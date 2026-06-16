from .base_policy import DispatchPolicy


class FCFSPolicy(DispatchPolicy):
    name = "fcfs"

    def dispatch(self, graph, idle_agvs, waiting_tasks):
        if not idle_agvs or not waiting_tasks:
            return None

        task = sorted(waiting_tasks, key=lambda t: (t.release_time, t.id))[0]
        agv = sorted(idle_agvs, key=lambda a: a.id)[0]

        return agv, task
