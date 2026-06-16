from __future__ import annotations

import simpy

from simulator.road_network import shortest_path_info


class SimulationEngine:
    def __init__(self, graph, agvs, tasks, policy, verbose: bool = True):
        self.graph = graph
        self.agvs = agvs
        self.tasks = sorted(tasks, key=lambda t: (t.release_time, t.id))
        self.policy = policy
        self.verbose = verbose

        self.env = simpy.Environment()
        self.waiting_tasks = []
        self.completed_tasks = []
        self.event_log = []
        self.travel_records = []

        self.dispatch_event = self.env.event()

    def log(self, message: str, **data):
        row = {
            "time": round(float(self.env.now), 3),
            "message": message,
        }
        row.update(data)
        self.event_log.append(row)

        if self.verbose:
            print(f"[{self.env.now:>7.1f} s] {message}")

    def add_travel_record(
        self,
        agv,
        task,
        path,
        distance,
        start_time,
        end_time,
        mode,
    ):
        self.travel_records.append(
            {
                "agv_id": agv.id,
                "task_id": task.id,
                "mode": mode,
                "path": path,
                "distance": round(float(distance), 3),
                "start_time": round(float(start_time), 3),
                "end_time": round(float(end_time), 3),
                "from_node": path[0],
                "to_node": path[-1],
            }
        )

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

            decision = self.policy.dispatch(self.graph, idle_agvs, waiting_tasks)

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

    def handle_task(self, agv, task):
        # 1. 空载去取货点
        path, distance = shortest_path_info(self.graph, agv.current_node, task.pickup_node)

        agv.status = "GO_TO_PICKUP"
        start_time = float(self.env.now)
        travel_time = distance / agv.speed_empty
        end_time = start_time + travel_time

        self.add_travel_record(
            agv=agv,
            task=task,
            path=path,
            distance=distance,
            start_time=start_time,
            end_time=end_time,
            mode="empty",
        )

        self.log(
            f"{agv.id} 空载前往 {task.pickup_node}，路径 {path}，距离 {distance:.1f}",
            event="GO_TO_PICKUP",
            agv_id=agv.id,
            task_id=task.id,
            distance=distance,
            mode="empty",
        )

        yield self.env.timeout(travel_time)

        agv.current_node = task.pickup_node
        agv.total_distance += distance
        agv.empty_distance += distance

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
        path, distance = shortest_path_info(self.graph, agv.current_node, task.dropoff_node)

        agv.status = "GO_TO_DROPOFF"
        start_time = float(self.env.now)
        travel_time = distance / agv.speed_loaded
        end_time = start_time + travel_time

        self.add_travel_record(
            agv=agv,
            task=task,
            path=path,
            distance=distance,
            start_time=start_time,
            end_time=end_time,
            mode="loaded",
        )

        self.log(
            f"{agv.id} 载货前往 {task.dropoff_node}，路径 {path}，距离 {distance:.1f}",
            event="GO_TO_DROPOFF",
            agv_id=agv.id,
            task_id=task.id,
            distance=distance,
            mode="loaded",
        )

        yield self.env.timeout(travel_time)

        agv.current_node = task.dropoff_node
        agv.total_distance += distance
        agv.loaded_distance += distance

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

        return {
            "policy": self.policy.name,
            "completed_tasks": len(self.completed_tasks),
            "makespan": round(max(task.finish_time for task in self.completed_tasks), 3),
            "avg_wait_time": round(sum(wait_times) / len(wait_times), 3),
            "avg_flow_time": round(sum(flow_times) / len(flow_times), 3),
            "total_distance": round(total_distance, 3),
            "empty_distance": round(empty_distance, 3),
            "loaded_distance": round(loaded_distance, 3),
            "empty_rate": round(empty_distance / total_distance, 3) if total_distance > 0 else 0.0,
        }
