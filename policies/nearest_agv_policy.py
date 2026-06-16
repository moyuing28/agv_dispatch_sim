import networkx as nx

from .base_policy import DispatchPolicy
from simulator.road_network import shortest_path_info


class NearestAGVPolicy(DispatchPolicy):
    name = "nearest"

    def dispatch(self, graph, idle_agvs, waiting_tasks):
        if not idle_agvs or not waiting_tasks:
            return None

        best = None

        for agv in idle_agvs:
            for task in waiting_tasks:
                try:
                    _, distance = shortest_path_info(graph, agv.current_node, task.pickup_node)
                except nx.NetworkXNoPath:
                    continue

                score = (distance, task.release_time, agv.id, task.id)

                if best is None or score < best[0]:
                    best = (score, agv, task)

        if best is None:
            return None

        return best[1], best[2]
