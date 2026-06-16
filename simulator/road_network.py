from __future__ import annotations

import importlib
import math
from types import ModuleType
from typing import Any

import networkx as nx


DEFAULT_MAP_MODULE = "configs.demo_map"


def make_line_points(start, end):
    """生成直线路径点。直线只允许横平竖直。"""
    x1, y1 = start
    x2, y2 = end

    if not (math.isclose(x1, x2, abs_tol=1e-9) or math.isclose(y1, y2, abs_tol=1e-9)):
        raise ValueError(
            f"直线边必须横平竖直，不能是斜线：start={start}, end={end}"
        )

    return [start, end]


def make_arc_points(center, radius, start_deg, end_deg, num=40):
    """根据圆心、半径、起止角度生成圆弧采样点。"""
    if radius <= 0:
        raise ValueError(f"圆弧半径必须大于 0：radius={radius}")

    angle_deg = abs(end_deg - start_deg)
    if not math.isclose(angle_deg, 90.0, abs_tol=1e-6):
        raise ValueError(
            f"目前只允许 90 度圆弧：start_deg={start_deg}, end_deg={end_deg}"
        )

    if num < 2:
        raise ValueError("圆弧采样点 num 至少为 2")

    cx, cy = center
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    points = []
    for i in range(num):
        t = i / (num - 1)
        angle = start_rad + t * (end_rad - start_rad)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    return points


def polyline_length(points):
    """计算折线近似长度。主要用于显示插值。"""
    total = 0.0
    for p1, p2 in zip(points[:-1], points[1:]):
        total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return total


def point_at_fraction(points, fraction=0.5):
    """按照显示几何的长度比例取点，用来放边长标签。"""
    if not points:
        raise ValueError("points 不能为空")
    if len(points) == 1:
        return points[0]

    total_len = polyline_length(points)
    if total_len <= 0:
        return points[0]

    target_len = total_len * fraction
    acc_len = 0.0

    for p1, p2 in zip(points[:-1], points[1:]):
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_len <= 0:
            continue

        if acc_len + seg_len >= target_len:
            ratio = (target_len - acc_len) / seg_len
            return (
                p1[0] + ratio * (p2[0] - p1[0]),
                p1[1] + ratio * (p2[1] - p1[1]),
            )

        acc_len += seg_len

    return points[-1]


def point_at_distance(points, distance):
    """按照显示几何的距离取点，用来做动画插值。"""
    if not points:
        raise ValueError("points 不能为空")
    if distance <= 0 or len(points) == 1:
        return points[0]

    acc_len = 0.0
    for p1, p2 in zip(points[:-1], points[1:]):
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_len <= 0:
            continue

        if acc_len + seg_len >= distance:
            ratio = (distance - acc_len) / seg_len
            return (
                p1[0] + ratio * (p2[0] - p1[0]),
                p1[1] + ratio * (p2[1] - p1[1]),
            )

        acc_len += seg_len

    return points[-1]


def is_close_point(p1, p2, eps=1e-6):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < eps


def load_map_config(map_module: str | ModuleType = DEFAULT_MAP_MODULE):
    """加载地图配置模块。默认读取 configs/demo_map.py。"""
    if isinstance(map_module, str):
        return importlib.import_module(map_module)
    return map_module


def _edge_points_and_length(positions: dict[str, tuple[float, float]], edge: dict[str, Any]):
    u = str(edge["u"])
    v = str(edge["v"])
    kind = edge["kind"]

    if u not in positions:
        raise KeyError(f"边 {u}->{v} 的起点 {u} 不在 positions 里")
    if v not in positions:
        raise KeyError(f"边 {u}->{v} 的终点 {v} 不在 positions 里")

    if kind == "line":
        points = make_line_points(positions[u], positions[v])
        length = math.hypot(
            positions[v][0] - positions[u][0],
            positions[v][1] - positions[u][1],
        )

    elif kind == "arc":
        points = make_arc_points(
            center=edge["center"],
            radius=float(edge["radius"]),
            start_deg=float(edge["start_deg"]),
            end_deg=float(edge["end_deg"]),
            num=int(edge.get("num", 40)),
        )

        if not is_close_point(points[0], positions[u]):
            raise ValueError(
                f"{u}->{v} 的圆弧起点 {points[0]} 和节点 {u} 坐标 {positions[u]} 不一致"
            )
        if not is_close_point(points[-1], positions[v]):
            raise ValueError(
                f"{u}->{v} 的圆弧终点 {points[-1]} 和节点 {v} 坐标 {positions[v]} 不一致"
            )

        angle_rad = abs(math.radians(float(edge["end_deg"]) - float(edge["start_deg"])))
        length = float(edge["radius"]) * angle_rad

    else:
        raise ValueError(f"未知边类型：{kind}。现在只允许 line 或 arc。")

    return u, v, points, float(length)


def build_graph_from_defs(positions, edge_defs):
    """
    根据手写 positions 和 edge_defs 建立图。

    graph: 给最短路径、调度仿真使用。
    edge_geometry: 给路网绘图、动画插值使用。

    length = 几何真实长度，用于图片/动图标注。
    weight = 算法权重，目前默认等于 length，后续可加入限速、转弯惩罚等因素。
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(str(node_id) for node_id in positions.keys())

    edge_geometry = {}

    for edge in edge_defs:
        u, v, points, length = _edge_points_and_length(positions, edge)
        weight = float(edge.get("weight", length))

        graph.add_edge(u, v, length=round(length, 3), weight=round(weight, 3))
        edge_geometry[(u, v)] = points

        # 默认双向。若以后有单行道，可在配置里写 bidirectional=False。
        if edge.get("bidirectional", True):
            graph.add_edge(v, u, length=round(length, 3), weight=round(weight, 3))
            edge_geometry[(v, u)] = list(reversed(points))

    return graph, edge_geometry


def build_demo_network(return_geometry: bool = False, map_module: str | ModuleType = DEFAULT_MAP_MODULE):
    """建立 AGV 路网。默认从 configs/demo_map.py 读取手写地图配置。"""
    config = load_map_config(map_module)
    positions = {str(k): tuple(v) for k, v in config.positions.items()}
    edge_defs = config.edge_defs

    graph, edge_geometry = build_graph_from_defs(positions, edge_defs)

    if return_geometry:
        return graph, positions, edge_geometry

    return graph, positions


def shortest_path_info(graph, source: str, target: str):
    """返回两个节点之间的最短路径和算法权重距离。"""
    path = nx.shortest_path(graph, source=str(source), target=str(target), weight="weight")
    distance = nx.shortest_path_length(graph, source=str(source), target=str(target), weight="weight")
    return path, float(distance)
