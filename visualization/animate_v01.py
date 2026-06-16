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


def get_agv_state_at_time(graph, positions, edge_geometry, records, agv_start_node, t):
    """根据轨迹记录，计算某辆 AGV 在 t 时刻的位置。"""
    current_position = positions[agv_start_node]
    current_label = "IDLE"

    for record in records:
        path = record["path"]
        start_time = float(record["start_time"])
        end_time = float(record["end_time"])
        mode = record["mode"]
        task_id = record["task_id"]

        if t < start_time:
            break

        if start_time <= t <= end_time:
            progress = 1.0 if end_time == start_time else (t - start_time) / (end_time - start_time)
            current_position = interpolate_on_path(
                graph=graph,
                positions=positions,
                edge_geometry=edge_geometry,
                path=path,
                progress=progress,
            )
            current_label = f"{mode}-{task_id}"
            return current_position, current_label

        if t > end_time:
            current_position = positions[path[-1]]
            current_label = "IDLE"

    return current_position, current_label


def draw_network(ax, graph, positions, edge_geometry, show_node_labels=True, show_edge_lengths=True):
    for u, v, data in graph.edges(data=True):
        if graph.has_edge(v, u) and str(u) > str(v):
            continue

        points = edge_geometry[(u, v)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=1.3, zorder=1)

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
                zorder=2,
            )

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_size=110,
        linewidths=0.5,
        edgecolors="black",
    )

    if show_node_labels:
        nx.draw_networkx_labels(
            graph,
            pos=positions,
            ax=ax,
            font_size=5,
            font_weight="bold",
        )


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

        draw_network(
            ax=ax,
            graph=graph,
            positions=positions,
            edge_geometry=edge_geometry,
            show_node_labels=not args.hide_node_labels,
            show_edge_lengths=not args.hide_edge_lengths,
        )

        for agv in agvs:
            pos, state_label = get_agv_state_at_time(
                graph=graph,
                positions=positions,
                edge_geometry=edge_geometry,
                records=agv_records[agv.id],
                agv_start_node=agv.start_node,
                t=t,
            )

            x, y = pos
            ax.scatter([x], [y], s=230, marker="s", zorder=5)
            ax.text(
                x,
                y + 0.45,
                f"{agv.id}\n{state_label}",
                ha="center",
                va="bottom",
                fontsize=7,
                zorder=6,
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
