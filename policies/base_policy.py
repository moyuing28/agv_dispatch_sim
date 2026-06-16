class DispatchPolicy:
    name = "base"

    def dispatch(self, graph, idle_agvs, waiting_tasks):
        """返回 (agv, task)，如果暂时不分配则返回 None。"""
        raise NotImplementedError
