import networkx as nx

from .base_policy import DispatchPolicy
from simulator.road_network import shortest_path_info


class NearestAGVPolicy(DispatchPolicy):
    name = "nearest"

    def dispatch(self, graph, idle_agvs, waiting_tasks, planner=None, now: float = 0.0):
        if not idle_agvs or not waiting_tasks:
            return None

        best = None

        for agv in idle_agvs:
            for task in waiting_tasks:
                try:
                    if planner is None:
                        # 兼容旧版本：没有 planner 时，退回原来的距离最短逻辑。
                        _, cost = shortest_path_info(graph, agv.current_node, task.pickup_node)
                        travel_time = cost / agv.speed_empty
                    else:
                        route = planner.plan(
                            graph,
                            agv.current_node,
                            task.pickup_node,
                            agv=agv,
                            mode="empty",
                            start_time=now,
                        )
                        cost = route.total_cost
                        travel_time = route.travel_time
                except nx.NetworkXNoPath:
                    continue

                score = (cost, travel_time, task.release_time, agv.id, task.id)

                if best is None or score < best[0]:
                    best = (score, agv, task)

        if best is None:
            return None

        return best[1], best[2]
