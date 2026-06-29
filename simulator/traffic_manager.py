from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simulator.models import RoutePlan


@dataclass
class ReservationResult:
    """一次路径预约的结果。"""

    route: object
    planned_start_time: float
    reserved_start_time: float
    wait_time: float
    conflict_count: int
    selected_candidate_index: int = 0
    candidate_count: int = 1
    reroute_used: bool = False


@dataclass
class _StepwiseBuildResult:
    route: RoutePlan
    internal_wait_time: float
    conflict_count: int
    conflict_records: list[dict]
    wait_records: list[dict]


class _RestartStepwisePlanning(Exception):
    """逐边预约过程中发现当前节点无法继续等待，需要整体推迟路线起点。"""

    def __init__(self, wait_until: float, conflict_records: list[dict]):
        super().__init__(f"需要整体推迟到 {wait_until:.3f}")
        self.wait_until = float(wait_until)
        self.conflict_records = conflict_records


class TrafficManager:
    """路径资源预约表。

    支持三种策略：
    1. whole_route：整条路径整体预约，冲突则出发前整体等待；
    2. stepwise：固定路径逐边预约，冲突则在当前节点等待；
    3. reroute：候选路径逐边预约，比较“原路等待”和“绕路”，选择实际到达最早的方案。

    Step 7 重点：
    - 节点容量 = 1；边容量 = 1；
    - 等待、装载、卸载都会占用节点；
    - reroute 模式不会只盯着最短路径，而是生成 K 条候选路径，逐条模拟严格预约后的真实时间，
      最后选择 end_time 最小的路径。
    """

    def __init__(
        self,
        safety_time: float = 0.5,
        node_hold_time: float = 0.5,
        max_iterations: int = 200,
        strategy: str = "whole_route",
        candidate_path_count: int = 5,
    ):
        if safety_time < 0:
            raise ValueError("safety_time 不能小于 0")
        if node_hold_time < 0:
            raise ValueError("node_hold_time 不能小于 0")
        if max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0")
        if strategy not in {"whole_route", "stepwise", "reroute"}:
            raise ValueError("strategy 只能是 whole_route、stepwise 或 reroute")
        if candidate_path_count <= 0:
            raise ValueError("candidate_path_count 必须大于 0")

        self.safety_time = float(safety_time)
        self.node_hold_time = float(node_hold_time)
        self.max_iterations = int(max_iterations)
        self.strategy = strategy
        self.mode = "reservation" if strategy == "whole_route" else strategy
        self.candidate_path_count = int(candidate_path_count)

        self.edge_reservations: list[dict] = []
        self.node_reservations: list[dict] = []
        self.conflict_records: list[dict] = []
        self.wait_records: list[dict] = []
        self.reroute_records: list[dict] = []

        self.total_wait_time = 0.0
        self.total_reserved_routes = 0
        self.total_conflict_count = 0
        self.total_reroute_checks = 0
        self.total_reroute_used = 0

    @staticmethod
    def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
        return max(float(start_a), float(start_b)) < min(float(end_a), float(end_b))

    @staticmethod
    def _target_occupancy_type(mode: str, target_hold_time: float) -> str:
        if target_hold_time <= 0:
            return "route_node"
        if mode == "empty":
            return "loading"
        if mode == "loaded":
            return "unloading"
        return "operation_hold"

    def _node_interval_from_schedule(self, node_row: dict) -> tuple[float, float]:
        arrival_time = float(node_row.get("arrival_time", 0.0))
        leave_time = float(node_row.get("leave_time", arrival_time))
        occupy_end_time = max(leave_time, arrival_time + self.node_hold_time)
        return arrival_time, occupy_end_time

    def _apply_target_hold(self, route, target_hold_time: float, mode: str):
        target_hold_time = max(0.0, float(target_hold_time))
        if not getattr(route, "node_schedules", None):
            return route

        final_node = route.node_schedules[-1]
        arrival_time = float(final_node.get("arrival_time", route.end_time))
        leave_time = max(float(final_node.get("leave_time", arrival_time)), arrival_time + target_hold_time)
        final_node["leave_time"] = round(leave_time, 3)
        final_node["operation_hold_time"] = round(target_hold_time, 3)
        final_node["occupancy_type"] = self._target_occupancy_type(mode, target_hold_time)
        return route

    def plan_and_reserve(
        self,
        *,
        planner,
        graph,
        source: str,
        target: str,
        agv,
        task,
        mode: str,
        start_time: float,
        target_hold_time: float = 0.0,
    ) -> ReservationResult:
        if self.strategy == "stepwise":
            return self._plan_and_reserve_stepwise(
                planner=planner,
                graph=graph,
                source=source,
                target=target,
                agv=agv,
                task=task,
                mode=mode,
                start_time=start_time,
                target_hold_time=target_hold_time,
            )

        if self.strategy == "reroute":
            return self._plan_and_reserve_reroute(
                planner=planner,
                graph=graph,
                source=source,
                target=target,
                agv=agv,
                task=task,
                mode=mode,
                start_time=start_time,
                target_hold_time=target_hold_time,
            )

        return self._plan_and_reserve_whole_route(
            planner=planner,
            graph=graph,
            source=source,
            target=target,
            agv=agv,
            task=task,
            mode=mode,
            start_time=start_time,
            target_hold_time=target_hold_time,
        )

    def _plan_and_reserve_whole_route(
        self,
        *,
        planner,
        graph,
        source: str,
        target: str,
        agv,
        task,
        mode: str,
        start_time: float,
        target_hold_time: float = 0.0,
    ) -> ReservationResult:
        planned_start_time = float(start_time)
        candidate_start_time = planned_start_time
        conflict_count = 0

        for _ in range(self.max_iterations):
            route = planner.plan(
                graph,
                source,
                target,
                agv=agv,
                mode=mode,
                start_time=candidate_start_time,
            )
            route = self._apply_target_hold(route, target_hold_time, mode)

            conflict = self.find_first_conflict(route, ignore_agv_id=agv.id)

            if conflict is None:
                wait_time = max(0.0, candidate_start_time - planned_start_time)
                self.reserve_route(
                    route=route,
                    agv_id=agv.id,
                    task_id=task.id,
                    mode=mode,
                    planned_start_time=planned_start_time,
                    wait_time=wait_time,
                    conflict_count=conflict_count,
                )
                self.total_wait_time += wait_time
                self.total_reserved_routes += 1

                return ReservationResult(
                    route=route,
                    planned_start_time=planned_start_time,
                    reserved_start_time=float(route.start_time),
                    wait_time=wait_time,
                    conflict_count=conflict_count,
                )

            conflict_count += 1
            self.total_conflict_count += 1

            conflict_record = {
                "strategy": self.strategy,
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "waiting_node": str(source),
                "edge_index": "",
                "planned_start_time": round(planned_start_time, 3),
                "candidate_start_time": round(candidate_start_time, 3),
                "new_candidate_start_time": round(float(conflict["wait_until"]), 3),
                "conflict_count_for_route": conflict_count,
            }
            conflict_record.update(conflict)
            self.conflict_records.append(conflict_record)

            next_start_time = float(conflict["wait_until"])
            if next_start_time <= candidate_start_time:
                next_start_time = candidate_start_time + max(self.safety_time, 1e-3)
            candidate_start_time = next_start_time

        raise RuntimeError(
            "路径预约失败：超过最大迭代次数。"
            f" agv={agv.id}, task={task.id}, mode={mode}, source={source}, target={target}"
        )

    def _plan_and_reserve_stepwise(
        self,
        *,
        planner,
        graph,
        source: str,
        target: str,
        agv,
        task,
        mode: str,
        start_time: float,
        target_hold_time: float = 0.0,
    ) -> ReservationResult:
        planned_start_time = float(start_time)
        candidate_start_time = planned_start_time
        all_conflict_count = 0

        for _ in range(self.max_iterations):
            nominal_route = planner.plan(
                graph,
                source,
                target,
                agv=agv,
                mode=mode,
                start_time=candidate_start_time,
            )

            try:
                build = self._build_stepwise_route(
                    nominal_route=nominal_route,
                    agv_id=agv.id,
                    task_id=task.id,
                    mode=mode,
                    route_start_time=candidate_start_time,
                    original_planned_start_time=planned_start_time,
                    target_hold_time=target_hold_time,
                )
            except _RestartStepwisePlanning as restart:
                self.conflict_records.extend(restart.conflict_records)
                self.total_conflict_count += len(restart.conflict_records)
                all_conflict_count += len(restart.conflict_records)
                next_start_time = max(candidate_start_time + max(self.safety_time, 1e-3), restart.wait_until)
                candidate_start_time = next_start_time
                continue

            pre_route_wait_time = max(0.0, candidate_start_time - planned_start_time)
            total_wait_time = pre_route_wait_time + build.internal_wait_time
            all_conflict_count += build.conflict_count

            self.conflict_records.extend(build.conflict_records)
            self.wait_records.extend(build.wait_records)
            self.total_conflict_count += build.conflict_count
            self._commit_route(
                route=build.route,
                agv_id=agv.id,
                task_id=task.id,
                mode=mode,
                planned_start_time=planned_start_time,
                wait_time=total_wait_time,
                conflict_count=all_conflict_count,
            )

            return ReservationResult(
                route=build.route,
                planned_start_time=planned_start_time,
                reserved_start_time=float(build.route.start_time),
                wait_time=total_wait_time,
                conflict_count=all_conflict_count,
            )

        raise RuntimeError(
            "逐边预约失败：当前节点容量约束导致多次整体推迟后仍不可行。"
            f" agv={agv.id}, task={task.id}, mode={mode}, source={source}, target={target}"
        )

    def _plan_and_reserve_reroute(
        self,
        *,
        planner,
        graph,
        source: str,
        target: str,
        agv,
        task,
        mode: str,
        start_time: float,
        target_hold_time: float = 0.0,
    ) -> ReservationResult:
        """候选路径选择：比较多条路径严格预约后的实际到达时间。"""
        planned_start_time = float(start_time)
        candidate_start_time = planned_start_time
        restart_conflict_count = 0

        for _ in range(self.max_iterations):
            nominal_routes = planner.plan_candidates(
                graph=graph,
                source=source,
                target=target,
                agv=agv,
                mode=mode,
                start_time=candidate_start_time,
                max_candidates=self.candidate_path_count,
            )
            if not nominal_routes:
                raise RuntimeError(f"无可用路径：agv={agv.id}, task={task.id}, source={source}, target={target}")

            evaluated: list[dict] = []
            restart_wait_untils: list[float] = []
            restart_records: list[dict] = []

            base_path = list(nominal_routes[0].path)

            for candidate_index, nominal_route in enumerate(nominal_routes):
                try:
                    build = self._build_stepwise_route(
                        nominal_route=nominal_route,
                        agv_id=agv.id,
                        task_id=task.id,
                        mode=mode,
                        route_start_time=candidate_start_time,
                        original_planned_start_time=planned_start_time,
                        target_hold_time=target_hold_time,
                    )
                except _RestartStepwisePlanning as restart:
                    restart_wait_untils.append(float(restart.wait_until))
                    restart_records.extend(restart.conflict_records)
                    self.reroute_records.append(
                        {
                            "strategy": self.strategy,
                            "agv_id": agv.id,
                            "task_id": task.id,
                            "mode": mode,
                            "candidate_index": candidate_index,
                            "candidate_status": "restart_required",
                            "candidate_path": nominal_route.path,
                            "candidate_distance": round(float(nominal_route.total_distance), 3),
                            "candidate_cost": round(float(nominal_route.total_cost), 3),
                            "candidate_start_time": round(candidate_start_time, 3),
                            "candidate_end_time": "",
                            "candidate_wait_time": "",
                            "candidate_conflict_count": len(restart.conflict_records),
                            "restart_wait_until": round(float(restart.wait_until), 3),
                            "selected": False,
                        }
                    )
                    continue

                internal_wait_time = build.internal_wait_time
                pre_route_wait_time = max(0.0, candidate_start_time - planned_start_time)
                total_wait_time = pre_route_wait_time + internal_wait_time
                reroute_used = list(build.route.path) != base_path

                evaluated.append(
                    {
                        "candidate_index": candidate_index,
                        "build": build,
                        "pre_route_wait_time": pre_route_wait_time,
                        "total_wait_time": total_wait_time,
                        "reroute_used": reroute_used,
                    }
                )

            if evaluated:
                # 选择实际到达最早，其次等待最少，再其次距离短、空间代价小。
                evaluated.sort(
                    key=lambda item: (
                        float(item["build"].route.end_time),
                        float(item["total_wait_time"]),
                        float(item["build"].route.total_distance),
                        float(item["build"].route.total_cost),
                        int(item["candidate_index"]),
                    )
                )
                best = evaluated[0]
                build: _StepwiseBuildResult = best["build"]
                total_wait_time = float(best["total_wait_time"])
                all_conflict_count = restart_conflict_count + build.conflict_count
                selected_candidate_index = int(best["candidate_index"])
                reroute_used = bool(best["reroute_used"])

                build.route.reroute_candidate_index = selected_candidate_index
                build.route.reroute_candidate_count = len(nominal_routes)
                build.route.reroute_used = reroute_used
                build.route.original_shortest_path = base_path

                self.conflict_records.extend(build.conflict_records)
                self.wait_records.extend(build.wait_records)
                self.total_conflict_count += build.conflict_count
                self.total_reroute_checks += 1
                if reroute_used:
                    self.total_reroute_used += 1

                self._commit_route(
                    route=build.route,
                    agv_id=agv.id,
                    task_id=task.id,
                    mode=mode,
                    planned_start_time=planned_start_time,
                    wait_time=total_wait_time,
                    conflict_count=all_conflict_count,
                )

                # 记录所有成功候选的评估结果，方便看“为什么选这条”。
                for item in evaluated:
                    cand_build: _StepwiseBuildResult = item["build"]
                    self.reroute_records.append(
                        {
                            "strategy": self.strategy,
                            "agv_id": agv.id,
                            "task_id": task.id,
                            "mode": mode,
                            "candidate_index": int(item["candidate_index"]),
                            "candidate_status": "feasible",
                            "candidate_path": cand_build.route.path,
                            "candidate_distance": round(float(cand_build.route.total_distance), 3),
                            "candidate_cost": round(float(cand_build.route.total_cost), 3),
                            "candidate_start_time": round(float(cand_build.route.start_time), 3),
                            "candidate_end_time": round(float(cand_build.route.end_time), 3),
                            "candidate_wait_time": round(float(item["total_wait_time"]), 3),
                            "candidate_internal_wait_time": round(float(cand_build.internal_wait_time), 3),
                            "candidate_conflict_count": int(cand_build.conflict_count),
                            "restart_wait_until": "",
                            "selected": int(item["candidate_index"]) == selected_candidate_index,
                        }
                    )

                return ReservationResult(
                    route=build.route,
                    planned_start_time=planned_start_time,
                    reserved_start_time=float(build.route.start_time),
                    wait_time=total_wait_time,
                    conflict_count=all_conflict_count,
                    selected_candidate_index=selected_candidate_index,
                    candidate_count=len(nominal_routes),
                    reroute_used=reroute_used,
                )

            # 没有任何候选路径能在当前起点安全排程：整体推迟后再试。
            if not restart_wait_untils:
                raise RuntimeError(
                    "reroute 模式失败：没有可行候选，也没有可用于推迟的冲突时间。"
                    f" agv={agv.id}, task={task.id}, mode={mode}, source={source}, target={target}"
                )
            self.conflict_records.extend(restart_records)
            self.total_conflict_count += len(restart_records)
            restart_conflict_count += len(restart_records)
            candidate_start_time = max(max(restart_wait_untils), candidate_start_time + max(self.safety_time, 1e-3))

        raise RuntimeError(
            "reroute 预约失败：多次候选路径评估后仍不可行。"
            f" agv={agv.id}, task={task.id}, mode={mode}, source={source}, target={target}"
        )

    def _commit_route(
        self,
        *,
        route,
        agv_id: str,
        task_id: str,
        mode: str,
        planned_start_time: float,
        wait_time: float,
        conflict_count: int,
    ):
        self.reserve_route(
            route=route,
            agv_id=agv_id,
            task_id=task_id,
            mode=mode,
            planned_start_time=planned_start_time,
            wait_time=wait_time,
            conflict_count=conflict_count,
        )
        self.total_wait_time += float(wait_time)
        self.total_reserved_routes += 1

    def _build_stepwise_route(
        self,
        *,
        nominal_route,
        agv_id: str,
        task_id: str,
        mode: str,
        route_start_time: float,
        original_planned_start_time: float,
        target_hold_time: float = 0.0,
    ) -> _StepwiseBuildResult:
        path = list(nominal_route.path)
        current_time = float(route_start_time)
        total_internal_wait = 0.0
        conflict_records: list[dict] = []
        wait_records: list[dict] = []
        edge_schedules: list[dict] = []
        node_schedules: list[dict] = []

        target_hold_time = max(0.0, float(target_hold_time))
        target_occupancy_type = self._target_occupancy_type(mode, target_hold_time)

        if len(path) == 1:
            leave_time = current_time + target_hold_time
            only_node_row = {
                "node": path[0],
                "node_index": 0,
                "arrival_time": round(current_time, 3),
                "leave_time": round(leave_time, 3),
                "wait_time": 0.0,
                "operation_hold_time": round(target_hold_time, 3),
                "occupancy_type": target_occupancy_type,
                "is_wait_node": False,
                "is_start": True,
                "is_end": True,
            }
            conflict = self._find_node_conflict(only_node_row, ignore_agv_id=agv_id)
            if conflict is not None:
                record = self._make_conflict_record(
                    conflict=conflict,
                    agv_id=agv_id,
                    task_id=task_id,
                    mode=mode,
                    waiting_node=path[0],
                    edge_index="",
                    planned_start_time=original_planned_start_time,
                    candidate_start_time=current_time,
                    candidate_end_time=leave_time,
                    new_candidate_start_time=float(conflict["wait_until"]),
                    conflict_count_for_route="",
                )
                raise _RestartStepwisePlanning(float(conflict["wait_until"]), [record])

            node_schedules.append(only_node_row)
            route = RoutePlan(
                path=path,
                total_distance=0.0,
                travel_time=0.0,
                total_cost=0.0,
                start_time=float(route_start_time),
                end_time=float(route_start_time),
                edge_distances=[],
                edge_times=[],
                edge_costs=[],
                edge_schedules=[],
                node_schedules=node_schedules,
                cost_model=nominal_route.cost_model,
            )
            route.wait_records = []
            return _StepwiseBuildResult(route=route, internal_wait_time=0.0, conflict_count=0, conflict_records=[], wait_records=[])

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
                "is_end": False,
            }
        )

        for edge_index, edge_template in enumerate(nominal_route.edge_schedules):
            u = edge_template["from_node"]
            v = edge_template["to_node"]
            edge_time = float(edge_template["travel_time"])
            is_last_edge = edge_index == len(nominal_route.edge_schedules) - 1
            destination_hold_time = target_hold_time if is_last_edge else 0.0

            edge_start_time, edge_end_time, conflicts_for_edge = self._find_earliest_edge_start(
                edge_template=edge_template,
                destination_node=v,
                destination_hold_time=destination_hold_time,
                edge_index=edge_index,
                earliest_start=current_time,
                edge_time=edge_time,
                agv_id=agv_id,
                task_id=task_id,
                mode=mode,
                planned_start_time=original_planned_start_time,
            )
            conflict_records.extend(conflicts_for_edge)

            wait_before_edge = max(0.0, edge_start_time - current_time)
            if wait_before_edge > 0:
                current_wait_node_row = {
                    "node": u,
                    "arrival_time": round(current_time, 3),
                    "leave_time": round(edge_start_time, 3),
                }
                current_node_conflict = self._find_node_conflict(current_wait_node_row, ignore_agv_id=agv_id)
                if current_node_conflict is not None:
                    conflict = dict(current_node_conflict)
                    conflict["conflict_type"] = "CURRENT_NODE_WAIT_CONFLICT"
                    record = self._make_conflict_record(
                        conflict=conflict,
                        agv_id=agv_id,
                        task_id=task_id,
                        mode=mode,
                        waiting_node=u,
                        edge_index=edge_index,
                        planned_start_time=original_planned_start_time,
                        candidate_start_time=current_time,
                        candidate_end_time=edge_start_time,
                        new_candidate_start_time=float(conflict["wait_until"]),
                        conflict_count_for_route="",
                    )
                    raise _RestartStepwisePlanning(float(conflict["wait_until"]), [record])

                wait_row = {
                    "strategy": self.strategy,
                    "agv_id": agv_id,
                    "task_id": task_id,
                    "mode": mode,
                    "node": u,
                    "node_index": edge_index,
                    "before_edge_key": edge_template["edge_key"],
                    "wait_start_time": round(current_time, 3),
                    "wait_end_time": round(edge_start_time, 3),
                    "wait_time": round(wait_before_edge, 3),
                    "conflict_count_for_wait": len(conflicts_for_edge),
                }
                wait_records.append(wait_row)
                total_internal_wait += wait_before_edge

            node_schedules[-1]["leave_time"] = round(edge_start_time, 3)
            node_schedules[-1]["wait_time"] = round(wait_before_edge, 3)
            node_schedules[-1]["is_wait_node"] = wait_before_edge > 0
            node_schedules[-1]["occupancy_type"] = "waiting" if wait_before_edge > 0 else node_schedules[-1].get("occupancy_type", "route_node")

            edge_row = dict(edge_template)
            edge_row.update(
                {
                    "start_time": round(edge_start_time, 3),
                    "end_time": round(edge_end_time, 3),
                    "wait_before_edge": round(wait_before_edge, 3),
                }
            )
            edge_schedules.append(edge_row)

            current_time = edge_end_time
            is_last_node = is_last_edge
            node_leave_time = current_time + (target_hold_time if is_last_node else 0.0)
            node_schedules.append(
                {
                    "node": v,
                    "node_index": edge_index + 1,
                    "arrival_time": round(current_time, 3),
                    "leave_time": round(node_leave_time, 3),
                    "wait_time": 0.0,
                    "operation_hold_time": round(target_hold_time if is_last_node else 0.0, 3),
                    "occupancy_type": target_occupancy_type if is_last_node else "route_node",
                    "is_wait_node": False,
                    "is_start": False,
                    "is_end": is_last_node,
                }
            )

        route = RoutePlan(
            path=path,
            total_distance=float(nominal_route.total_distance),
            travel_time=float(current_time - route_start_time),
            total_cost=float(nominal_route.total_cost),
            start_time=float(route_start_time),
            end_time=float(current_time),
            edge_distances=list(nominal_route.edge_distances),
            edge_times=list(nominal_route.edge_times),
            edge_costs=list(nominal_route.edge_costs),
            edge_schedules=edge_schedules,
            node_schedules=node_schedules,
            cost_model=nominal_route.cost_model,
        )
        route.wait_records = wait_records
        return _StepwiseBuildResult(
            route=route,
            internal_wait_time=total_internal_wait,
            conflict_count=len(conflict_records),
            conflict_records=conflict_records,
            wait_records=wait_records,
        )

    def _find_earliest_edge_start(
        self,
        *,
        edge_template: dict,
        destination_node: str,
        destination_hold_time: float,
        edge_index: int,
        earliest_start: float,
        edge_time: float,
        agv_id: str,
        task_id: str,
        mode: str,
        planned_start_time: float,
    ) -> tuple[float, float, list[dict]]:
        edge_start = float(earliest_start)
        conflict_records: list[dict] = []
        destination_hold_time = max(0.0, float(destination_hold_time))

        for _ in range(self.max_iterations):
            edge_end = edge_start + edge_time
            candidate_edge = dict(edge_template)
            candidate_edge.update({"start_time": round(edge_start, 3), "end_time": round(edge_end, 3)})
            destination_node_row = {
                "node": destination_node,
                "arrival_time": round(edge_end, 3),
                "leave_time": round(edge_end + destination_hold_time, 3),
            }

            conflicts = []
            edge_conflict = self._find_edge_conflict(candidate_edge, ignore_agv_id=agv_id)
            if edge_conflict is not None:
                conflicts.append(edge_conflict)
            node_conflict = self._find_node_conflict(destination_node_row, ignore_agv_id=agv_id)
            if node_conflict is not None:
                conflicts.append(node_conflict)

            if not conflicts:
                return edge_start, edge_end, conflict_records

            new_edge_start = edge_start
            for conflict in conflicts:
                if conflict["resource_type"] == "edge":
                    proposed_start = float(conflict["wait_until"])
                else:
                    proposed_start = float(conflict["wait_until"]) - edge_time - destination_hold_time
                new_edge_start = max(new_edge_start, proposed_start)

                record = self._make_conflict_record(
                    conflict=conflict,
                    agv_id=agv_id,
                    task_id=task_id,
                    mode=mode,
                    waiting_node=edge_template["from_node"],
                    edge_index=edge_index,
                    planned_start_time=planned_start_time,
                    candidate_start_time=edge_start,
                    candidate_end_time=edge_end,
                    new_candidate_start_time=new_edge_start,
                    conflict_count_for_route="",
                )
                conflict_records.append(record)

            if new_edge_start <= edge_start:
                new_edge_start = edge_start + max(self.safety_time, 1e-3)
            edge_start = new_edge_start

        raise RuntimeError(
            "逐边预约失败：超过最大迭代次数。"
            f" agv={agv_id}, task={task_id}, mode={mode}, edge={edge_template.get('edge_key')}"
        )

    def _make_conflict_record(
        self,
        *,
        conflict: dict,
        agv_id: str,
        task_id: str,
        mode: str,
        waiting_node,
        edge_index,
        planned_start_time: float,
        candidate_start_time: float,
        candidate_end_time: float,
        new_candidate_start_time: float,
        conflict_count_for_route,
    ) -> dict:
        record = {
            "strategy": self.strategy,
            "agv_id": agv_id,
            "task_id": task_id,
            "mode": mode,
            "waiting_node": str(waiting_node),
            "edge_index": edge_index,
            "planned_start_time": round(float(planned_start_time), 3),
            "candidate_start_time": round(float(candidate_start_time), 3),
            "candidate_end_time": round(float(candidate_end_time), 3),
            "new_candidate_start_time": round(float(new_candidate_start_time), 3),
            "conflict_count_for_route": conflict_count_for_route,
        }
        record.update(conflict)
        return record

    def find_first_conflict(self, route, ignore_agv_id: str | None = None) -> Optional[dict]:
        conflicts: list[dict] = []
        for edge_row in route.edge_schedules:
            edge_conflict = self._find_edge_conflict(edge_row, ignore_agv_id=ignore_agv_id)
            if edge_conflict is not None:
                conflicts.append(edge_conflict)
        for node_row in route.node_schedules:
            node_conflict = self._find_node_conflict(node_row, ignore_agv_id=ignore_agv_id)
            if node_conflict is not None:
                conflicts.append(node_conflict)
        if not conflicts:
            return None
        conflicts.sort(key=lambda item: (float(item["conflict_start_time"]), float(item["wait_until"])))
        return conflicts[0]

    def _find_edge_conflict(self, edge_row: dict, ignore_agv_id: str | None = None) -> Optional[dict]:
        candidate_start = float(edge_row["start_time"])
        candidate_end = float(edge_row["end_time"])
        candidate_edge_key = edge_row["edge_key"]
        candidate_undirected_key = edge_row["undirected_edge_key"]
        found: list[dict] = []

        for reservation in self.edge_reservations:
            if ignore_agv_id is not None and reservation.get("agv_id") == ignore_agv_id:
                continue
            if reservation["undirected_edge_key"] != candidate_undirected_key:
                continue
            reserved_start = float(reservation["start_time"])
            reserved_end_with_buffer = float(reservation["end_time"]) + self.safety_time
            if not self._overlap(candidate_start, candidate_end, reserved_start, reserved_end_with_buffer):
                continue
            conflict_type = "EDGE_SAME_DIRECTION" if reservation["edge_key"] == candidate_edge_key else "EDGE_OPPOSITE_DIRECTION"
            found.append(
                {
                    "resource_type": "edge",
                    "conflict_type": conflict_type,
                    "resource_key": candidate_undirected_key,
                    "candidate_edge_key": candidate_edge_key,
                    "existing_edge_key": reservation["edge_key"],
                    "candidate_start_time": round(candidate_start, 3),
                    "candidate_end_time": round(candidate_end, 3),
                    "existing_start_time": round(reserved_start, 3),
                    "existing_end_time": round(float(reservation["end_time"]), 3),
                    "existing_agv_id": reservation["agv_id"],
                    "existing_task_id": reservation["task_id"],
                    "existing_mode": reservation["mode"],
                    "conflict_start_time": round(max(candidate_start, reserved_start), 3),
                    "conflict_end_time": round(min(candidate_end, reserved_end_with_buffer), 3),
                    "wait_until": round(reserved_end_with_buffer, 3),
                }
            )
        if not found:
            return None
        found.sort(key=lambda item: (float(item["conflict_start_time"]), float(item["wait_until"])))
        return found[0]

    def _find_node_conflict(self, node_row: dict, ignore_agv_id: str | None = None) -> Optional[dict]:
        candidate_node = str(node_row["node"])
        candidate_start, candidate_end = self._node_interval_from_schedule(node_row)
        found: list[dict] = []

        for reservation in self.node_reservations:
            if ignore_agv_id is not None and reservation.get("agv_id") == ignore_agv_id:
                continue
            if str(reservation["node"]) != candidate_node:
                continue
            reserved_start = float(reservation["occupy_start_time"])
            reserved_end_with_buffer = float(reservation["occupy_end_time"]) + self.safety_time
            if not self._overlap(candidate_start, candidate_end, reserved_start, reserved_end_with_buffer):
                continue
            found.append(
                {
                    "resource_type": "node",
                    "conflict_type": "NODE_CONFLICT",
                    "resource_key": candidate_node,
                    "candidate_edge_key": "",
                    "existing_edge_key": "",
                    "candidate_start_time": round(candidate_start, 3),
                    "candidate_end_time": round(candidate_end, 3),
                    "existing_start_time": round(reserved_start, 3),
                    "existing_end_time": round(float(reservation["occupy_end_time"]), 3),
                    "existing_agv_id": reservation["agv_id"],
                    "existing_task_id": reservation["task_id"],
                    "existing_mode": reservation["mode"],
                    "existing_occupancy_type": reservation.get("occupancy_type", ""),
                    "conflict_start_time": round(max(candidate_start, reserved_start), 3),
                    "conflict_end_time": round(min(candidate_end, reserved_end_with_buffer), 3),
                    "wait_until": round(reserved_end_with_buffer, 3),
                }
            )
        if not found:
            return None
        found.sort(key=lambda item: (float(item["conflict_start_time"]), float(item["wait_until"])))
        return found[0]

    def reserve_route(
        self,
        *,
        route,
        agv_id: str,
        task_id: str,
        mode: str,
        planned_start_time: float,
        wait_time: float,
        conflict_count: int,
    ):
        reservation_id = f"RES{self.total_reserved_routes + 1:05d}_{agv_id}_{task_id}_{mode}"

        for edge_row in route.edge_schedules:
            row = {
                "reservation_id": reservation_id,
                "strategy": self.strategy,
                "agv_id": agv_id,
                "task_id": task_id,
                "mode": mode,
                "planned_start_time": round(float(planned_start_time), 3),
                "reserved_start_time": round(float(route.start_time), 3),
                "wait_time": round(float(wait_time), 3),
                "conflict_count": int(conflict_count),
                "reroute_candidate_index": getattr(route, "reroute_candidate_index", 0),
                "reroute_used": getattr(route, "reroute_used", False),
            }
            row.update(edge_row)
            self.edge_reservations.append(row)

        for node_row in route.node_schedules:
            occupy_start_time, occupy_end_time = self._node_interval_from_schedule(node_row)
            row = {
                "reservation_id": reservation_id,
                "strategy": self.strategy,
                "agv_id": agv_id,
                "task_id": task_id,
                "mode": mode,
                "planned_start_time": round(float(planned_start_time), 3),
                "reserved_start_time": round(float(route.start_time), 3),
                "wait_time": round(float(wait_time), 3),
                "conflict_count": int(conflict_count),
                "occupy_start_time": round(float(occupy_start_time), 3),
                "occupy_end_time": round(float(occupy_end_time), 3),
                "occupancy_type": node_row.get("occupancy_type", "route_node"),
                "reroute_candidate_index": getattr(route, "reroute_candidate_index", 0),
                "reroute_used": getattr(route, "reroute_used", False),
            }
            row.update(node_row)
            self.node_reservations.append(row)

    def summary(self) -> dict:
        loading_node_reservations = sum(1 for row in self.node_reservations if row.get("occupancy_type") == "loading")
        unloading_node_reservations = sum(1 for row in self.node_reservations if row.get("occupancy_type") == "unloading")
        waiting_node_reservations = sum(1 for row in self.node_reservations if row.get("occupancy_type") == "waiting")
        return {
            "traffic_strategy": self.strategy,
            "traffic_safety_time": self.safety_time,
            "traffic_node_hold_time": self.node_hold_time,
            "traffic_node_capacity": 1,
            "traffic_edge_capacity": 1,
            "traffic_candidate_path_count": self.candidate_path_count,
            "traffic_reserved_routes": self.total_reserved_routes,
            "traffic_total_wait_time": round(float(self.total_wait_time), 3),
            "traffic_conflict_count": self.total_conflict_count,
            "traffic_edge_reservation_count": len(self.edge_reservations),
            "traffic_node_reservation_count": len(self.node_reservations),
            "traffic_wait_record_count": len(self.wait_records),
            "traffic_reroute_check_count": self.total_reroute_checks,
            "traffic_reroute_used_count": self.total_reroute_used,
            "traffic_reroute_record_count": len(self.reroute_records),
            "traffic_loading_node_reservation_count": loading_node_reservations,
            "traffic_unloading_node_reservation_count": unloading_node_reservations,
            "traffic_waiting_node_reservation_count": waiting_node_reservations,
        }
