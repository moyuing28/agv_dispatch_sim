from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_v01 import build_agvs
from simulator.road_network import build_demo_network, point_at_distance, point_at_fraction, polyline_length


# =========================
# 显示颜色配置
# =========================
ROAD_COLOR = "#1f77b4"          # 统一道路颜色：蓝色
ROAD_LINE_WIDTH = 1.4

HIGHLIGHT_COLOR = "#ffbf00"     # 规划未来路径高亮：黄色/琥珀色，和道路蓝色明显区分
HIGHLIGHT_LINE_WIDTH = 4.2

NODE_COLOR = "#f2f2f2"

# AGV 颜色不要用蓝色系，避免和道路颜色接近。
AGV_COLOR_PALETTE = [
    "#d62728",  # 红
    "#2ca02c",  # 绿
    "#9467bd",  # 紫
    "#ff7f0e",  # 橙
    "#8c564b",  # 棕
    "#e377c2",  # 粉
    "#bcbd22",  # 橄榄黄
    "#7f7f7f",  # 灰
]


def get_agv_color(agv_id: str, agv_index: int) -> str:
    """按 AGV 顺序分配固定颜色，保证每帧颜色不变。"""
    return AGV_COLOR_PALETTE[agv_index % len(AGV_COLOR_PALETTE)]


def interpolate_on_path(graph, positions, edge_geometry, path, progress):
    """根据 progress 在真实几何路径上插值。"""
    if len(path) == 1:
        return positions[path[0]]

    edge_lengths = [float(graph[u][v]["length"]) for u, v in zip(path[:-1], path[1:])]
    total_length = sum(edge_lengths)

    if total_length <= 0:
        return positions[path[-1]]

    target_distance = progress * total_length
    traveled = 0.0

    for i, length in enumerate(edge_lengths):
        u = path[i]
        v = path[i + 1]

        if traveled + length >= target_distance:
            remain = target_distance - traveled
            points = edge_geometry[(u, v)]
            display_len = polyline_length(points)
            if length <= 0:
                return points[-1]
            return point_at_distance(points, display_len * remain / length)

        traveled += length

    return positions[path[-1]]


def remaining_points_from_distance(points, distance):
    """
    返回一条边上从 distance 位置到终点的剩余几何点。
    用于实现：AGV 走过的路恢复普通蓝色，只高亮未来还没走的路。
    """
    if not points:
        return []

    if distance <= 0 or len(points) == 1:
        return list(points)

    total_len = polyline_length(points)
    if distance >= total_len:
        return [points[-1]]

    acc_len = 0.0
    for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
        seg_len = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if seg_len <= 0:
            continue

        if acc_len + seg_len >= distance:
            ratio = (distance - acc_len) / seg_len
            current_point = (
                p1[0] + ratio * (p2[0] - p1[0]),
                p1[1] + ratio * (p2[1] - p1[1]),
            )
            return [current_point] + list(points[i + 1:])

        acc_len += seg_len

    return [points[-1]]


def draw_remaining_planned_path(ax, graph, edge_geometry, path, progress):
    """
    高亮 AGV 当前记录中“未来还要走”的路径。

    progress 表示当前 travel record 已完成比例：
    - 已经走过的边不再高亮，因为底图会重新画成普通蓝色；
    - 当前所在边只高亮当前位置到该边终点；
    - 后续边整段高亮。
    """
    if not path or len(path) < 2:
        return

    progress = max(0.0, min(1.0, float(progress)))
    edge_lengths = [float(graph[u][v]["length"]) for u, v in zip(path[:-1], path[1:])]
    total_length = sum(edge_lengths)

    if total_length <= 0:
        return

    target_distance = progress * total_length
    traveled = 0.0

    for i, edge_length in enumerate(edge_lengths):
        u = path[i]
        v = path[i + 1]
        edge_start = traveled
        edge_end = traveled + edge_length
        points = edge_geometry[(u, v)]

        # 这条边已经完全走过，不画高亮。
        if target_distance >= edge_end:
            traveled = edge_end
            continue

        # 这条边还没开始走，整条边高亮。
        if target_distance <= edge_start:
            remaining_points = points
        else:
            # AGV 正在这条边上，只画当前位置到终点的剩余部分。
            display_len = polyline_length(points)
            if edge_length <= 0:
                remaining_points = [points[-1]]
            else:
                distance_on_display_edge = display_len * (target_distance - edge_start) / edge_length
                remaining_points = remaining_points_from_distance(points, distance_on_display_edge)

        if len(remaining_points) >= 2:
            xs = [p[0] for p in remaining_points]
            ys = [p[1] for p in remaining_points]
            ax.plot(
                xs,
                ys,
                color=HIGHLIGHT_COLOR,
                linewidth=HIGHLIGHT_LINE_WIDTH,
                solid_capstyle="round",
                zorder=3,
            )

        traveled = edge_end


def get_agv_state_at_time(graph, positions, edge_geometry, records, agv_start_node, t):
    """根据轨迹记录，计算某辆 AGV 在 t 时刻的位置、状态和当前正在执行的路径。"""
    current_position = positions[agv_start_node]
    current_label = "IDLE"
    active_record = None
    active_progress = 0.0

    for record in records:
        path = record["path"]
        start_time = float(record["start_time"])
        end_time = float(record["end_time"])
        mode = record["mode"]
        task_id = record["task_id"]

        if t < start_time:
            break

        if start_time <= t <= end_time:
            active_progress = 1.0 if end_time == start_time else (t - start_time) / (end_time - start_time)
            current_position = interpolate_on_path(
                graph=graph,
                positions=positions,
                edge_geometry=edge_geometry,
                path=path,
                progress=active_progress,
            )
            current_label = f"{mode}-{task_id}"
            active_record = record
            return current_position, current_label, active_record, active_progress

        if t > end_time:
            current_position = positions[path[-1]]
            current_label = "IDLE"

    return current_position, current_label, active_record, active_progress


def draw_network(ax, graph, positions, edge_geometry, show_node_labels=True, show_edge_lengths=True):
    for u, v, data in graph.edges(data=True):
        if graph.has_edge(v, u) and str(u) > str(v):
            continue

        points = edge_geometry[(u, v)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            color=ROAD_COLOR,
            linewidth=ROAD_LINE_WIDTH,
            solid_capstyle="round",
            zorder=1,
        )

        if show_edge_lengths:
            label_x, label_y = point_at_fraction(points, fraction=0.5)
            ax.text(
                label_x,
                label_y,
                f'{data["length"]:.1f}',
                fontsize=5,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
                zorder=4,
            )

    node_collection = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_size=110,
        node_color=NODE_COLOR,
        linewidths=0.5,
        edgecolors="black",
    )
    node_collection.set_zorder(4)

    if show_node_labels:
        label_artists = nx.draw_networkx_labels(
            graph,
            pos=positions,
            ax=ax,
            font_size=5,
            font_weight="bold",
        )
        for label in label_artists.values():
            label.set_zorder(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["fcfs", "nearest"],
        default="nearest",
        help="选择要可视化的策略",
    )
    parser.add_argument(
        "--map",
        default="configs.demo_map",
        help="地图配置模块，例如 configs.demo_map",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="动画帧间隔，单位毫秒",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
        help="仿真时间采样步长，单位秒",
    )
    parser.add_argument(
        "--hide-node-labels",
        action="store_true",
        help="动画中隐藏节点编号",
    )
    parser.add_argument(
        "--hide-edge-lengths",
        action="store_true",
        help="动画中隐藏边的真实长度",
    )

    args = parser.parse_args()

    graph, positions, edge_geometry = build_demo_network(return_geometry=True, map_module=args.map)
    agvs = build_agvs()

    travel_path = PROJECT_ROOT / "outputs" / f"v01_travels_{args.policy}.csv"

    if not travel_path.exists():
        raise FileNotFoundError(
            f"找不到轨迹文件：{travel_path}\n"
            f"请先运行：python experiments/run_v01.py --policy {args.policy}"
        )

    df = pd.read_csv(travel_path)

    if df.empty:
        raise RuntimeError("轨迹文件为空，无法生成动画。")

    df["path"] = df["path"].apply(ast.literal_eval)

    agv_records = {}
    for agv in agvs:
        agv_records[agv.id] = (
            df[df["agv_id"] == agv.id]
            .sort_values("start_time")
            .to_dict("records")
        )

    max_time = float(df["end_time"].max())
    times = list(pd.Series([i * args.step for i in range(int(max_time / args.step) + 2)]))

    all_points = []
    for points in edge_geometry.values():
        all_points.extend(points)
    all_points.extend(positions.values())
    xs_all = [p[0] for p in all_points]
    ys_all = [p[1] for p in all_points]
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    pad_x = max((x_max - x_min) * 0.06, 1.0)
    pad_y = max((y_max - y_min) * 0.10, 1.0)

    fig, ax = plt.subplots(figsize=(14, 9))

    def update(frame_index):
        ax.clear()
        t = times[frame_index]

        # 每帧先画普通蓝色底图，所以 AGV 走过的道路会自动恢复为正常颜色。
        draw_network(
            ax=ax,
            graph=graph,
            positions=positions,
            edge_geometry=edge_geometry,
            show_node_labels=not args.hide_node_labels,
            show_edge_lengths=not args.hide_edge_lengths,
        )

        active_states = []
        for agv in agvs:
            pos, state_label, active_record, active_progress = get_agv_state_at_time(
                graph=graph,
                positions=positions,
                edge_geometry=edge_geometry,
                records=agv_records[agv.id],
                agv_start_node=agv.start_node,
                t=t,
            )
            active_states.append((agv, pos, state_label, active_record, active_progress))

        # 先画所有 AGV 的未来规划路径，再画 AGV 图标，避免高亮线盖住小车。
        for agv, _, _, active_record, active_progress in active_states:
            if active_record is not None:
                draw_remaining_planned_path(
                    ax=ax,
                    graph=graph,
                    edge_geometry=edge_geometry,
                    path=active_record["path"],
                    progress=active_progress,
                )

        for agv_index, (agv, pos, state_label, _, _) in enumerate(active_states):
            x, y = pos
            agv_color = get_agv_color(agv.id, agv_index)
            ax.scatter(
                [x],
                [y],
                s=230,
                marker="s",
                c=[agv_color],
                edgecolors="black",
                linewidths=0.9,
                zorder=6,
            )
            ax.text(
                x,
                y + 0.45,
                f"{agv.id}\n{state_label}",
                ha="center",
                va="bottom",
                fontsize=7,
                zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        ax.set_title(f"AGV Dispatch Animation | Policy: {args.policy} | t = {t:.1f} s")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=args.interval,
        repeat=True,
    )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"v01_animation_{args.policy}.gif"
    ani.save(output_file, writer="pillow", fps=max(1, int(1000 / args.interval)))
    plt.close(fig)

    print(f"动画已保存：{output_file}")


if __name__ == "__main__":
    main()
