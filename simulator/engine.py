from __future__ import annotations

import simpy

from simulator.route_planner import RoutePlanner


class SimulationEngine:
    def __init__(
        self,
        graph,
        agvs,
        tasks,
        policy,
        verbose: bool = True,
        route_planner=None,
        traffic_manager=None,
    ):
        self.graph = graph
        self.agvs = agvs
        self.tasks = sorted(tasks, key=lambda t: (t.release_time, t.id))
        self.policy = policy
        self.verbose = verbose
        self.route_planner = route_planner or RoutePlanner()
        self.traffic_manager = traffic_manager

        self.env = simpy.Environment()
        self.waiting_tasks = []
        self.completed_tasks = []
        self.event_log = []
        self.travel_records = []
        self.route_edge_records = []
        self.route_node_records = []
        self.route_wait_records = []

        self.dispatch_event = self.env.event()
        self._route_counter = 0

    def log(self, message: str, **data):
        row = {
            "time": round(float(self.env.now), 3),
            "message": message,
        }
        row.update(data)
        self.event_log.append(row)

        if self.verbose:
            print(f"[{self.env.now:>7.1f} s] {message}")

    def _next_route_id(self, agv, task, mode: str) -> str:
        self._route_counter += 1
        return f"R{self._route_counter:04d}_{agv.id}_{task.id}_{mode}"

    def add_travel_record(
        self,
        agv,
        task,
        route,
        planned_start_time,
        start_time,
        end_time,
        mode,
        traffic_wait_time: float = 0.0,
        pre_route_wait_time: float = 0.0,
        traffic_conflict_count: int = 0,
    ):
        route_id = self._next_route_id(agv=agv, task=task, mode=mode)
        wait_records = list(getattr(route, "wait_records", []))
        internal_wait_time = sum(float(row.get("wait_time", 0.0)) for row in wait_records)
        moving_time = sum(float(x) for x in route.edge_times)

        self.travel_records.append(
            {
                "route_id": route_id,
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "path": route.path,
                "distance": round(float(route.total_distance), 3),
                "moving_time": round(float(moving_time), 3),
                "travel_time": round(float(route.travel_time), 3),
                "route_cost": round(float(route.total_cost), 3),
                "cost_model": route.cost_model,
                "planned_start_time": round(float(planned_start_time), 3),
                "start_time": round(float(start_time), 3),
                "end_time": round(float(end_time), 3),
                "pre_route_wait_time": round(float(pre_route_wait_time), 3),
                "internal_wait_time": round(float(internal_wait_time), 3),
                "traffic_wait_time": round(float(traffic_wait_time), 3),
                "traffic_conflict_count": int(traffic_conflict_count),
                "from_node": route.path[0],
                "to_node": route.path[-1],
                "edge_count": len(route.edge_schedules),
                "node_count": len(route.node_schedules),
                "wait_record_count": len(wait_records),
                "reroute_used": getattr(route, "reroute_used", False),
                "reroute_candidate_index": getattr(route, "reroute_candidate_index", 0),
                "reroute_candidate_count": getattr(route, "reroute_candidate_count", 1),
                "original_shortest_path": getattr(route, "original_shortest_path", route.path),
            }
        )

        for edge_row in route.edge_schedules:
            row = {
                "route_id": route_id,
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "cost_model": route.cost_model,
                "pre_route_wait_time": round(float(pre_route_wait_time), 3),
                "internal_wait_time": round(float(internal_wait_time), 3),
                "traffic_wait_time": round(float(traffic_wait_time), 3),
                "traffic_conflict_count": int(traffic_conflict_count),
            }
            row.update(edge_row)
            self.route_edge_records.append(row)

        for node_row in route.node_schedules:
            row = {
                "route_id": route_id,
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "cost_model": route.cost_model,
                "pre_route_wait_time": round(float(pre_route_wait_time), 3),
                "internal_wait_time": round(float(internal_wait_time), 3),
                "traffic_wait_time": round(float(traffic_wait_time), 3),
                "traffic_conflict_count": int(traffic_conflict_count),
            }
            row.update(node_row)
            self.route_node_records.append(row)

        for wait_row in wait_records:
            row = {
                "route_id": route_id,
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "cost_model": route.cost_model,
            }
            row.update(wait_row)
            self.route_wait_records.append(row)

    def notify_dispatch(self):
        if not self.dispatch_event.triggered:
            self.dispatch_event.succeed()

    def task_generator(self):
        for task in self.tasks:
            delay = task.release_time - self.env.now

            if delay > 0:
                yield self.env.timeout(delay)

            self.waiting_tasks.append(task)

            self.log(
                f"任务 {task.id} 释放：{task.pickup_node} -> {task.dropoff_node}",
                event="TASK_RELEASED",
                task_id=task.id,
            )

            self.notify_dispatch()

    def dispatcher(self):
        while len(self.completed_tasks) < len(self.tasks):
            yield self.dispatch_event
            self.dispatch_event = self.env.event()
            self.dispatch_available_tasks()

    def dispatch_available_tasks(self):
        while True:
            idle_agvs = [agv for agv in self.agvs if agv.status == "IDLE"]
            waiting_tasks = [task for task in self.waiting_tasks if task.assigned_agv is None]

            if not idle_agvs or not waiting_tasks:
                return

            decision = self.policy.dispatch(
                self.graph,
                idle_agvs,
                waiting_tasks,
                planner=self.route_planner,
                now=float(self.env.now),
            )

            if decision is None:
                return

            agv, task = decision

            if agv.status != "IDLE" or task not in self.waiting_tasks:
                return

            self.waiting_tasks.remove(task)

            task.assigned_agv = agv.id
            task.start_time = float(self.env.now)

            agv.status = "ASSIGNED"
            agv.current_task_id = task.id

            self.log(
                f"调度：{agv.id} 执行 {task.id}",
                event="DISPATCH",
                agv_id=agv.id,
                task_id=task.id,
            )

            self.env.process(self.handle_task(agv, task))

    def _plan_route_with_optional_reservation(self, agv, task, source, target, mode: str, target_hold_time: float = 0.0):
        planned_start_time = float(self.env.now)

        if self.traffic_manager is None:
            route = self.route_planner.plan(
                self.graph,
                source,
                target,
                agv=agv,
                mode=mode,
                start_time=planned_start_time,
            )
            return route, planned_start_time, 0.0, 0.0, 0

        result = self.traffic_manager.plan_and_reserve(
            planner=self.route_planner,
            graph=self.graph,
            source=source,
            target=target,
            agv=agv,
            task=task,
            mode=mode,
            start_time=planned_start_time,
            target_hold_time=target_hold_time,
        )

        strategy = getattr(self.traffic_manager, "strategy", "whole_route")
        if strategy == "whole_route":
            pre_route_wait_time = result.wait_time
        else:
            # stepwise 中可能因为当前节点无法安全等待而整体推迟路线起点，
            # 这部分属于出发前等待；route.wait_records 只记录中途等待。
            pre_route_wait_time = max(0.0, float(result.reserved_start_time) - float(planned_start_time))

        return result.route, planned_start_time, result.wait_time, pre_route_wait_time, result.conflict_count

    def _wait_before_reserved_route(self, agv, task, route, mode: str, pre_route_wait_time: float):
        if pre_route_wait_time <= 0:
            return

        agv.status = "WAITING_FOR_ROUTE"
        self.log(
            f"{agv.id} 出发前等待道路资源 {pre_route_wait_time:.1f}s 后再执行 {task.id}-{mode}",
            event="TRAFFIC_PRE_ROUTE_WAIT",
            agv_id=agv.id,
            task_id=task.id,
            mode=mode,
            wait_time=round(float(pre_route_wait_time), 3),
            reserved_start_time=round(float(route.start_time), 3),
        )

    def _log_internal_waits(self, agv, task, route, mode: str):
        wait_records = list(getattr(route, "wait_records", []))
        for wait_row in wait_records:
            self.log(
                f"{agv.id} 计划在节点 {wait_row['node']} 中途等待 "
                f"{float(wait_row['wait_time']):.1f}s 后进入边 {wait_row['before_edge_key']}",
                event="TRAFFIC_INTERNAL_WAIT_PLANNED",
                agv_id=agv.id,
                task_id=task.id,
                mode=mode,
                node=wait_row["node"],
                before_edge_key=wait_row["before_edge_key"],
                wait_time=round(float(wait_row["wait_time"]), 3),
                wait_start_time=wait_row["wait_start_time"],
                wait_end_time=wait_row["wait_end_time"],
            )

    def _execute_route(self, agv, task, route, planned_start_time, traffic_wait_time, pre_route_wait_time, traffic_conflict_count, mode: str, target_node: str):
        self._wait_before_reserved_route(
            agv=agv,
            task=task,
            route=route,
            mode=mode,
            pre_route_wait_time=pre_route_wait_time,
        )
        if pre_route_wait_time > 0:
            yield self.env.timeout(pre_route_wait_time)

        if getattr(route, "wait_records", []):
            self._log_internal_waits(agv=agv, task=task, route=route, mode=mode)

        agv.status = "GO_TO_PICKUP" if mode == "empty" else "GO_TO_DROPOFF"
        start_time = float(self.env.now)
        travel_time = route.travel_time
        end_time = start_time + travel_time

        self.add_travel_record(
            agv=agv,
            task=task,
            route=route,
            planned_start_time=planned_start_time,
            start_time=start_time,
            end_time=end_time,
            mode=mode,
            traffic_wait_time=traffic_wait_time,
            pre_route_wait_time=pre_route_wait_time,
            traffic_conflict_count=traffic_conflict_count,
        )

        action_text = "空载前往" if mode == "empty" else "载货前往"
        self.log(
            f"{agv.id} {action_text} {target_node}，路径 {route.path}，"
            f"距离 {route.total_distance:.1f}，预计总耗时 {route.travel_time:.1f}",
            event="GO_TO_PICKUP" if mode == "empty" else "GO_TO_DROPOFF",
            agv_id=agv.id,
            task_id=task.id,
            distance=round(route.total_distance, 3),
            moving_time=round(sum(float(x) for x in route.edge_times), 3),
            travel_time=round(route.travel_time, 3),
            route_cost=round(route.total_cost, 3),
            reroute_used=getattr(route, "reroute_used", False),
            reroute_candidate_index=getattr(route, "reroute_candidate_index", 0),
            traffic_wait_time=round(float(traffic_wait_time), 3),
            pre_route_wait_time=round(float(pre_route_wait_time), 3),
            internal_wait_time=round(sum(float(row.get("wait_time", 0.0)) for row in getattr(route, "wait_records", [])), 3),
            traffic_conflict_count=int(traffic_conflict_count),
            cost_model=route.cost_model,
            mode=mode,
        )

        yield self.env.timeout(travel_time)

    def handle_task(self, agv, task):
        # 1. 空载去取货点
        route, planned_start_time, traffic_wait_time, pre_route_wait_time, traffic_conflict_count = self._plan_route_with_optional_reservation(
            agv=agv,
            task=task,
            source=agv.current_node,
            target=task.pickup_node,
            mode="empty",
            target_hold_time=task.loading_time,
        )

        yield from self._execute_route(
            agv=agv,
            task=task,
            route=route,
            planned_start_time=planned_start_time,
            traffic_wait_time=traffic_wait_time,
            pre_route_wait_time=pre_route_wait_time,
            traffic_conflict_count=traffic_conflict_count,
            mode="empty",
            target_node=task.pickup_node,
        )

        agv.current_node = task.pickup_node
        agv.total_distance += route.total_distance
        agv.empty_distance += route.total_distance

        # 2. 装载
        agv.status = "LOADING"

        self.log(
            f"{agv.id} 开始装载 {task.id}",
            event="LOADING",
            agv_id=agv.id,
            task_id=task.id,
        )

        yield self.env.timeout(task.loading_time)

        # 3. 载货去卸货点
        route, planned_start_time, traffic_wait_time, pre_route_wait_time, traffic_conflict_count = self._plan_route_with_optional_reservation(
            agv=agv,
            task=task,
            source=agv.current_node,
            target=task.dropoff_node,
            mode="loaded",
            target_hold_time=task.unloading_time,
        )

        yield from self._execute_route(
            agv=agv,
            task=task,
            route=route,
            planned_start_time=planned_start_time,
            traffic_wait_time=traffic_wait_time,
            pre_route_wait_time=pre_route_wait_time,
            traffic_conflict_count=traffic_conflict_count,
            mode="loaded",
            target_node=task.dropoff_node,
        )

        agv.current_node = task.dropoff_node
        agv.total_distance += route.total_distance
        agv.loaded_distance += route.total_distance

        # 4. 卸载
        agv.status = "UNLOADING"

        self.log(
            f"{agv.id} 开始卸载 {task.id}",
            event="UNLOADING",
            agv_id=agv.id,
            task_id=task.id,
        )

        yield self.env.timeout(task.unloading_time)

        # 5. 任务完成
        task.finish_time = float(self.env.now)

        agv.status = "IDLE"
        agv.current_task_id = None
        agv.completed_task_ids.append(task.id)

        self.completed_tasks.append(task)

        self.log(
            f"{agv.id} 完成 {task.id}",
            event="TASK_FINISHED",
            agv_id=agv.id,
            task_id=task.id,
        )

        self.notify_dispatch()

    def run(self):
        self.env.process(self.dispatcher())
        self.env.process(self.task_generator())
        self.env.run()

        return self.summary()

    def summary(self):
        if not self.completed_tasks:
            return {}

        wait_times = [
            task.start_time - task.release_time
            for task in self.completed_tasks
        ]

        flow_times = [
            task.finish_time - task.release_time
            for task in self.completed_tasks
        ]

        total_distance = sum(agv.total_distance for agv in self.agvs)
        empty_distance = sum(agv.empty_distance for agv in self.agvs)
        loaded_distance = sum(agv.loaded_distance for agv in self.agvs)
        traffic_wait_total = sum(float(row.get("traffic_wait_time", 0.0)) for row in self.travel_records)
        pre_route_wait_total = sum(float(row.get("pre_route_wait_time", 0.0)) for row in self.travel_records)
        internal_wait_total = sum(float(row.get("internal_wait_time", 0.0)) for row in self.travel_records)

        traffic_mode = "none"
        if self.traffic_manager is not None:
            traffic_mode = getattr(self.traffic_manager, "mode", "reservation")

        summary = {
            "policy": self.policy.name,
            "cost_model": self.route_planner.cost_model.name,
            "traffic_mode": traffic_mode,
            "completed_tasks": len(self.completed_tasks),
            "makespan": round(max(task.finish_time for task in self.completed_tasks), 3),
            "avg_wait_time": round(sum(wait_times) / len(wait_times), 3),
            "avg_flow_time": round(sum(flow_times) / len(flow_times), 3),
            "total_distance": round(total_distance, 3),
            "empty_distance": round(empty_distance, 3),
            "loaded_distance": round(loaded_distance, 3),
            "empty_rate": round(empty_distance / total_distance, 3) if total_distance > 0 else 0.0,
            "traffic_wait_total": round(float(traffic_wait_total), 3),
            "pre_route_wait_total": round(float(pre_route_wait_total), 3),
            "internal_wait_total": round(float(internal_wait_total), 3),
            "route_record_count": len(self.travel_records),
            "route_edge_record_count": len(self.route_edge_records),
            "route_node_record_count": len(self.route_node_records),
            "route_wait_record_count": len(self.route_wait_records),
        }

        if self.traffic_manager is not None:
            summary.update(self.traffic_manager.summary())

        return summary
