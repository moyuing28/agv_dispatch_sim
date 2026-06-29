class DispatchPolicy:
    name = "base"

    def dispatch(self, graph, idle_agvs, waiting_tasks, planner=None, now: float = 0.0):
        """返回 (agv, task)，如果暂时不分配则返回 None。

        planner 和 now 是为后续路径代价、时间窗冲突检测预留的接口。
        """
        raise NotImplementedError
