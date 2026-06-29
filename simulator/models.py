from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    id: str
    release_time: float
    pickup_node: str
    dropoff_node: str
    loading_time: float = 20.0
    unloading_time: float = 20.0
    priority: int = 1
    assigned_agv: Optional[str] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None


@dataclass
class AGV:
    id: str
    start_node: str
    speed_empty: float = 1.2
    speed_loaded: float = 1.0
    current_node: Optional[str] = None
    status: str = "IDLE"
    current_task_id: Optional[str] = None
    total_distance: float = 0.0
    empty_distance: float = 0.0
    loaded_distance: float = 0.0
    completed_task_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.current_node is None:
            self.current_node = self.start_node


@dataclass
class RoutePlan:
    """路径规划结果。

    path：节点序列。
    total_distance：真实几何距离，用于里程统计和动画显示。
    travel_time：预计通行时间，用于仿真推进。
    total_cost：路径规划代价，用于不同 cost model 比较。

    edge_schedules：路径中每条边的时间表，为后续边冲突/对向冲突检测服务。
    node_schedules：路径中每个节点的到达时间表，为后续节点冲突检测服务。
    """

    path: list[str]
    total_distance: float
    travel_time: float
    total_cost: float
    start_time: float = 0.0
    end_time: float = 0.0
    edge_distances: list[float] = field(default_factory=list)
    edge_times: list[float] = field(default_factory=list)
    edge_costs: list[float] = field(default_factory=list)
    edge_schedules: list[dict] = field(default_factory=list)
    node_schedules: list[dict] = field(default_factory=list)
    cost_model: str = "distance"
