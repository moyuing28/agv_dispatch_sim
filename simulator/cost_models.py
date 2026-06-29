from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseCostModel:
    """路径规划代价模型基类。

    设计原则：
    - length：真实几何距离，用于统计里程和可视化。
    - travel_time：预计通行时间，用于仿真推进。
    - cost：路径规划代价，用于最短路搜索。

    后续要考虑转弯、加减速、拥堵等待时，只需要扩展 cost 模型，
    不必再改 engine.py 的主流程。
    """

    name: str = "base"

    def edge_distance(self, edge_data: dict) -> float:
        return float(edge_data.get("length", edge_data.get("weight", 0.0)))

    def get_agv_speed(self, agv, mode: str) -> float:
        if agv is None:
            return 1.0

        if mode == "loaded":
            return float(agv.speed_loaded)

        return float(agv.speed_empty)

    def edge_travel_time(self, edge_data: dict, agv=None, mode: str = "empty") -> float:
        distance = self.edge_distance(edge_data)
        speed = self.get_agv_speed(agv, mode)

        if speed <= 0:
            raise ValueError(f"AGV 速度必须大于 0：speed={speed}")

        return distance / speed

    def edge_cost(self, edge_data: dict, agv=None, mode: str = "empty") -> float:
        raise NotImplementedError

    def parameters(self) -> dict:
        """返回模型参数，方便写入 summary CSV。"""
        return {}


@dataclass
class DistanceCostModel(BaseCostModel):
    """距离最短模型：与原始版本的路径规划逻辑基本一致。"""

    name: str = "distance"

    def edge_cost(self, edge_data: dict, agv=None, mode: str = "empty") -> float:
        return self.edge_distance(edge_data)


@dataclass
class TimeCostModel(BaseCostModel):
    """时间最短模型：用通行时间作为路径规划代价。

    第二阶段核心：
    - line 直线路段：默认使用 AGV 自身速度，或使用 line_speed_* 覆盖。
    - arc  弯道路段：默认使用 line_speed * arc_speed_ratio，体现过弯低速。
    - arc_delay：可选的每个弯道固定时间损失，用于模拟进出弯减速/调整。

    第一版建议只改 arc_speed_ratio，例如 0.5 表示弯道速度为直线速度的一半。
    """

    name: str = "time"
    line_speed_empty: Optional[float] = None
    line_speed_loaded: Optional[float] = None
    arc_speed_empty: Optional[float] = None
    arc_speed_loaded: Optional[float] = None
    arc_speed_ratio: float = 0.5
    arc_delay: float = 0.0

    def _line_speed(self, agv, mode: str) -> float:
        if mode == "loaded" and self.line_speed_loaded is not None:
            return float(self.line_speed_loaded)
        if mode != "loaded" and self.line_speed_empty is not None:
            return float(self.line_speed_empty)

        return self.get_agv_speed(agv, mode)

    def _arc_speed(self, agv, mode: str) -> float:
        if mode == "loaded" and self.arc_speed_loaded is not None:
            return float(self.arc_speed_loaded)
        if mode != "loaded" and self.arc_speed_empty is not None:
            return float(self.arc_speed_empty)

        line_speed = self._line_speed(agv=agv, mode=mode)
        return line_speed * float(self.arc_speed_ratio)

    def _configured_speed(self, edge_data: dict, agv, mode: str) -> float:
        edge_kind = edge_data.get("kind", "line")

        if edge_kind == "arc":
            return self._arc_speed(agv=agv, mode=mode)

        return self._line_speed(agv=agv, mode=mode)

    def edge_travel_time(self, edge_data: dict, agv=None, mode: str = "empty") -> float:
        distance = self.edge_distance(edge_data)
        speed = self._configured_speed(edge_data=edge_data, agv=agv, mode=mode)

        if speed <= 0:
            raise ValueError(f"AGV 速度必须大于 0：speed={speed}")

        travel_time = distance / speed

        if edge_data.get("kind", "line") == "arc":
            travel_time += float(self.arc_delay)

        return travel_time

    def edge_cost(self, edge_data: dict, agv=None, mode: str = "empty") -> float:
        return self.edge_travel_time(edge_data=edge_data, agv=agv, mode=mode)

    def parameters(self) -> dict:
        return {
            "line_speed_empty": self.line_speed_empty,
            "line_speed_loaded": self.line_speed_loaded,
            "arc_speed_empty": self.arc_speed_empty,
            "arc_speed_loaded": self.arc_speed_loaded,
            "arc_speed_ratio": self.arc_speed_ratio,
            "arc_delay": self.arc_delay,
        }


def get_cost_model(
    name: str,
    *,
    line_speed_empty: Optional[float] = None,
    line_speed_loaded: Optional[float] = None,
    arc_speed_empty: Optional[float] = None,
    arc_speed_loaded: Optional[float] = None,
    arc_speed_ratio: float = 0.5,
    arc_delay: float = 0.0,
) -> BaseCostModel:
    """根据名称创建代价模型。"""
    normalized = name.lower().strip()

    if normalized == "distance":
        return DistanceCostModel()

    if normalized == "time":
        return TimeCostModel(
            line_speed_empty=line_speed_empty,
            line_speed_loaded=line_speed_loaded,
            arc_speed_empty=arc_speed_empty,
            arc_speed_loaded=arc_speed_loaded,
            arc_speed_ratio=arc_speed_ratio,
            arc_delay=arc_delay,
        )

    raise ValueError(f"未知 cost model：{name}")
