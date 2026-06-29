from __future__ import annotations

import itertools

import networkx as nx

from simulator.cost_models import BaseCostModel, DistanceCostModel
from simulator.models import RoutePlan


class RoutePlanner:
    """统一路径规划入口。

    Step 7 新增：
    - plan()：保持原有接口，返回一条最优路径；
    - plan_candidates()：返回多条候选路径，供 TrafficManager 比较“原路等待”和“绕路”。

    注意：这里仍然只负责生成空间路径和名义时间表，真正的边/节点预约、等待、绕路选择
    仍然交给 TrafficManager。
    """

    def __init__(self, cost_model: BaseCostModel | None = None):
        self.cost_model = cost_model or DistanceCostModel()

    def plan(
        self,
        graph,
        source: str,
        target: str,
        agv=None,
        mode: str = "empty",
        start_time: float = 0.0,
    ) -> RoutePlan:
        candidates = self.plan_candidates(
            graph=graph,
            source=source,
            target=target,
            agv=agv,
            mode=mode,
            start_time=start_time,
            max_candidates=1,
        )
        if not candidates:
            raise nx.NetworkXNoPath(f"No path between {source} and {target}")
        return candidates[0]

    def plan_candidates(
        self,
        graph,
        source: str,
        target: str,
        agv=None,
        mode: str = "empty",
        start_time: float = 0.0,
        max_candidates: int = 5,
    ) -> list[RoutePlan]:
        """返回若干条按 cost 排序的简单路径候选。

        max_candidates=1 时等价于普通最短路。候选路径用于后续 TrafficManager 评估：
        某条路径虽然空间 cost 稍大，但如果避开已预约节点/边，总完成时间可能更短。
        """
        source = str(source)
        target = str(target)
        start_time = float(start_time)
        max_candidates = max(1, int(max_candidates))

        def weight_func(u, v, edge_data):
            return self.cost_model.edge_cost(edge_data=edge_data, agv=agv, mode=mode)

        if source == target:
            return [
                self._build_route_from_path(
                    graph=graph,
                    path=[source],
                    agv=agv,
                    mode=mode,
                    start_time=start_time,
                )
            ]

        try:
            path_iter = nx.shortest_simple_paths(
                graph,
                source=source,
                target=target,
                weight=weight_func,
            )
            raw_paths = list(itertools.islice(path_iter, max_candidates))
        except nx.NetworkXNoPath:
            return []

        return [
            self._build_route_from_path(
                graph=graph,
                path=path,
                agv=agv,
                mode=mode,
                start_time=start_time,
            )
            for path in raw_paths
        ]

    def _build_route_from_path(
        self,
        graph,
        path: list[str],
        agv=None,
        mode: str = "empty",
        start_time: float = 0.0,
    ) -> RoutePlan:
        path = [str(node) for node in path]
        start_time = float(start_time)

        edge_distances: list[float] = []
        edge_times: list[float] = []
        edge_costs: list[float] = []
        edge_schedules: list[dict] = []
        node_schedules: list[dict] = []

        current_time = start_time

        if len(path) == 1:
            node_schedules.append(
                {
                    "node": path[0],
                    "node_index": 0,
                    "arrival_time": round(current_time, 3),
                    "leave_time": round(current_time, 3),
                    "wait_time": 0.0,
                    "operation_hold_time": 0.0,
                    "occupancy_type": "route_node",
                    "is_wait_node": False,
                    "is_start": True,
                    "is_end": True,
                }
            )
            return RoutePlan(
                path=path,
                total_distance=0.0,
                travel_time=0.0,
                total_cost=0.0,
                start_time=start_time,
                end_time=start_time,
                edge_distances=edge_distances,
                edge_times=edge_times,
                edge_costs=edge_costs,
                edge_schedules=edge_schedules,
                node_schedules=node_schedules,
                cost_model=self.cost_model.name,
            )

        node_schedules.append(
            {
                "node": path[0],
                "node_index": 0,
                "arrival_time": round(start_time, 3),
                "leave_time": round(start_time, 3),
                "wait_time": 0.0,
                "operation_hold_time": 0.0,
                "occupancy_type": "route_node",
                "is_wait_node": False,
                "is_start": True,
                "is_end": False,
            }
        )

        for edge_index, (u, v) in enumerate(zip(path[:-1], path[1:])):
            edge_data = graph[u][v]
            distance = self.cost_model.edge_distance(edge_data)
            edge_time = self.cost_model.edge_travel_time(edge_data, agv=agv, mode=mode)
            edge_cost = self.cost_model.edge_cost(edge_data, agv=agv, mode=mode)
            edge_start_time = current_time
            edge_end_time = edge_start_time + edge_time

            edge_distances.append(float(distance))
            edge_times.append(float(edge_time))
            edge_costs.append(float(edge_cost))

            edge_schedules.append(
                {
                    "edge_index": edge_index,
                    "from_node": str(u),
                    "to_node": str(v),
                    "edge_key": f"{u}->{v}",
                    "undirected_edge_key": "-".join(sorted([str(u), str(v)])),
                    "kind": edge_data.get("kind", "line"),
                    "distance": round(float(distance), 3),
                    "travel_time": round(float(edge_time), 3),
                    "route_cost": round(float(edge_cost), 3),
                    "start_time": round(float(edge_start_time), 3),
                    "end_time": round(float(edge_end_time), 3),
                }
            )

            is_last_node = edge_index == len(path) - 2
            node_schedules.append(
                {
                    "node": str(v),
                    "node_index": edge_index + 1,
                    "arrival_time": round(float(edge_end_time), 3),
                    "leave_time": round(float(edge_end_time), 3),
                    "wait_time": 0.0,
                    "operation_hold_time": 0.0,
                    "occupancy_type": "route_node",
                    "is_wait_node": False,
                    "is_start": False,
                    "is_end": is_last_node,
                }
            )

            current_time = edge_end_time

        total_distance = sum(edge_distances)
        travel_time = sum(edge_times)
        total_cost = sum(edge_costs)
        end_time = start_time + travel_time

        return RoutePlan(
            path=path,
            total_distance=float(total_distance),
            travel_time=float(travel_time),
            total_cost=float(total_cost),
            start_time=float(start_time),
            end_time=float(end_time),
            edge_distances=edge_distances,
            edge_times=edge_times,
            edge_costs=edge_costs,
            edge_schedules=edge_schedules,
            node_schedules=node_schedules,
            cost_model=self.cost_model.name,
        )
