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
