from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pandas.errors import EmptyDataError

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


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        return run_dir

    latest_path = PROJECT_ROOT / "outputs" / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"找不到 {latest_path}\n"
            "请先运行 experiments/run_v01.py，或手动指定 --run-dir outputs/runs/某个实验目录。"
        )

    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)

    run_dir = Path(latest["run_dir"])
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return run_dir


def read_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {"run_id": run_dir.name, "map": "configs.demo_map"}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """读取 CSV；文件不存在或为空时返回空 DataFrame。"""
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_agv_color(agv_id: str, agv_index: int) -> str:
    """按 AGV 顺序分配固定颜色，保证每帧颜色不变。"""
    return AGV_COLOR_PALETTE[agv_index % len(AGV_COLOR_PALETTE)]


def edge_position(edge_geometry, u: str, v: str, progress: float):
    """根据 progress 在一条真实几何边上插值。"""
    progress = max(0.0, min(1.0, float(progress)))
    points = edge_geometry[(str(u), str(v))]
    display_len = polyline_length(points)
    return point_at_distance(points, display_len * progress)


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


def draw_edge_points(ax, points):
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(
        xs,
        ys,
        color=HIGHLIGHT_COLOR,
        linewidth=HIGHLIGHT_LINE_WIDTH,
        solid_capstyle="round",
        zorder=3,
    )


def draw_remaining_edge(ax, edge_geometry, u: str, v: str, progress: float):
    """高亮当前边从 AGV 当前位置到终点的剩余部分。"""
    points = edge_geometry[(str(u), str(v))]
    display_len = polyline_length(points)
    remaining = remaining_points_from_distance(points, display_len * max(0.0, min(1.0, progress)))
    draw_edge_points(ax, remaining)


def draw_full_edge(ax, edge_geometry, u: str, v: str):
    """高亮完整边。"""
    points = edge_geometry[(str(u), str(v))]
    draw_edge_points(ax, points)


def make_edge_event(row: dict) -> dict:
    return {
        "event_type": "edge",
        "route_id": row.get("route_id", ""),
        "agv_id": row.get("agv_id", ""),
        "task_id": row.get("task_id", ""),
        "mode": row.get("mode", ""),
        "start_time": to_float(row.get("start_time")),
        "end_time": to_float(row.get("end_time")),
        "from_node": str(row.get("from_node")),
        "to_node": str(row.get("to_node")),
        "edge_index": int(to_float(row.get("edge_index"), 0)),
        "edge_key": row.get("edge_key", ""),
    }


def make_wait_event(row: dict) -> dict:
    return {
        "event_type": "wait",
        "route_id": row.get("route_id", ""),
        "agv_id": row.get("agv_id", ""),
        "task_id": row.get("task_id", ""),
        "mode": row.get("mode", ""),
        "start_time": to_float(row.get("wait_start_time")),
        "end_time": to_float(row.get("wait_end_time")),
        "node": str(row.get("node")),
        "node_index": int(to_float(row.get("node_index"), 0)),
        "before_edge_key": row.get("before_edge_key", ""),
    }


def build_agv_events(edge_df: pd.DataFrame, wait_df: pd.DataFrame, agvs) -> dict[str, list[dict]]:
    """把 route_edges 和 route_waits 合并成每辆 AGV 的时间事件序列。"""
    agv_events = {agv.id: [] for agv in agvs}

    if not edge_df.empty:
        for row in edge_df.to_dict("records"):
            agv_id = row.get("agv_id")
            if agv_id in agv_events:
                agv_events[agv_id].append(make_edge_event(row))

    if not wait_df.empty:
        for row in wait_df.to_dict("records"):
            agv_id = row.get("agv_id")
            if agv_id in agv_events:
                event = make_wait_event(row)
                # wait_time 为 0 的记录不需要显示。
                if event["end_time"] > event["start_time"]:
                    agv_events[agv_id].append(event)

    for agv_id, events in agv_events.items():
        # 同一时刻先显示 wait，再显示 edge；正常情况下 wait_end == edge_start，下一帧进入边。
        events.sort(key=lambda e: (e["start_time"], e["end_time"], 0 if e["event_type"] == "wait" else 1))

    return agv_events


def build_route_edges(edge_df: pd.DataFrame) -> dict[str, list[dict]]:
    """按 route_id 组织边记录，用于绘制当前路线的剩余高亮。"""
    route_edges: dict[str, list[dict]] = {}
    if edge_df.empty:
        return route_edges

    for row in edge_df.to_dict("records"):
        route_id = row.get("route_id", "")
        if not route_id:
            continue
        edge = make_edge_event(row)
        route_edges.setdefault(route_id, []).append(edge)

    for edges in route_edges.values():
        edges.sort(key=lambda e: (e["edge_index"], e["start_time"], e["end_time"]))

    return route_edges


def get_agv_state_at_time(positions, edge_geometry, events: list[dict], agv_start_node: str, t: float):
    """
    根据 route_edges + route_waits 计算某辆 AGV 在 t 时刻的位置和状态。

    edge 事件：AGV 在边上移动；
    wait 事件：AGV 停在等待节点；
    没有活动事件时，保持在上一条边的终点或起始节点。
    """
    current_position = positions[agv_start_node]
    current_label = "IDLE"
    active_event = None

    for event in events:
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])

        if t < start_time:
            break

        if start_time <= t <= end_time:
            task_id = event.get("task_id", "")
            mode = event.get("mode", "")

            if event["event_type"] == "edge":
                duration = end_time - start_time
                progress = 1.0 if duration <= 0 else (t - start_time) / duration
                current_position = edge_position(
                    edge_geometry=edge_geometry,
                    u=event["from_node"],
                    v=event["to_node"],
                    progress=progress,
                )
                current_label = f"{mode}-{task_id}"
                active_event = dict(event)
                active_event["progress"] = progress
                return current_position, current_label, active_event

            if event["event_type"] == "wait":
                current_position = positions[event["node"]]
                current_label = f"WAIT-{mode}-{task_id}"
                active_event = dict(event)
                active_event["progress"] = 0.0
                return current_position, current_label, active_event

        if t > end_time:
            if event["event_type"] == "edge":
                current_position = positions[event["to_node"]]
            elif event["event_type"] == "wait":
                current_position = positions[event["node"]]
            current_label = "IDLE"

    return current_position, current_label, active_event


def draw_remaining_route(ax, edge_geometry, route_edges: dict[str, list[dict]], active_event: dict | None, t: float):
    """高亮当前 route_id 中还没走完的边。"""
    if active_event is None:
        return

    route_id = active_event.get("route_id", "")
    if not route_id or route_id not in route_edges:
        return

    for edge in route_edges[route_id]:
        edge_start = float(edge["start_time"])
        edge_end = float(edge["end_time"])

        # 这条边已经走完，不再高亮。
        if edge_end < t:
            continue

        # 正在这条边上，画当前位置到终点。
        if active_event["event_type"] == "edge" and edge.get("edge_key") == active_event.get("edge_key"):
            draw_remaining_edge(
                ax=ax,
                edge_geometry=edge_geometry,
                u=edge["from_node"],
                v=edge["to_node"],
                progress=float(active_event.get("progress", 0.0)),
            )
            continue

        # 当前处于等待状态，或者这条边还没开始走：整条边都是未来路径。
        if edge_start >= t:
            draw_full_edge(
                ax=ax,
                edge_geometry=edge_geometry,
                u=edge["from_node"],
                v=edge["to_node"],
            )


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


def get_max_time(*dfs: pd.DataFrame) -> float:
    candidates: list[float] = []
    for df in dfs:
        if df.empty:
            continue
        for col in ["end_time", "wait_end_time", "finish_time"]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not values.empty:
                    candidates.append(float(values.max()))
    return max(candidates) if candidates else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help="实验输出目录，例如 outputs/runs/v01_nearest_time_stepwise。默认读取 outputs/latest_run.json。",
    )
    parser.add_argument(
        "--policy",
        choices=["fcfs", "nearest"],
        default=None,
        help="兼容旧版本参数；新版本优先使用 --run-dir 或 latest_run.json。",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="地图配置模块。若不指定，优先从 run_dir/config.json 读取。",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="动画帧间隔，单位毫秒",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
        help="仿真时间采样步长，单位秒",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="动画输出路径。若不指定，保存到 run_dir/figures/animation.gif。",
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

    run_dir = resolve_run_dir(args.run_dir)
    config = read_run_config(run_dir)
    map_module = args.map or config.get("map") or config.get("args", {}).get("map") or "configs.demo_map"
    run_id = config.get("run_id", run_dir.name)

    graph, positions, edge_geometry = build_demo_network(return_geometry=True, map_module=map_module)
    agvs = build_agvs()

    edge_path = run_dir / "route_edges.csv"
    wait_path = run_dir / "route_waits.csv"
    travel_path = run_dir / "travels.csv"

    edge_df = read_csv_or_empty(edge_path)
    wait_df = read_csv_or_empty(wait_path)
    travel_df = read_csv_or_empty(travel_path)

    if edge_df.empty:
        raise FileNotFoundError(
            f"找不到可用的逐边轨迹文件：{edge_path}\n"
            "请用 normal 或 debug 输出级别重新运行实验，例如：\n"
            "python experiments/run_v01.py --policy nearest --cost-model time --traffic-mode stepwise --output-level normal --quiet"
        )

    required_edge_cols = {"agv_id", "route_id", "from_node", "to_node", "start_time", "end_time"}
    missing = required_edge_cols - set(edge_df.columns)
    if missing:
        raise RuntimeError(f"route_edges.csv 缺少必要列：{sorted(missing)}")

    agv_events = build_agv_events(edge_df=edge_df, wait_df=wait_df, agvs=agvs)
    route_edges = build_route_edges(edge_df=edge_df)

    max_time = get_max_time(edge_df, wait_df, travel_df)
    if max_time <= 0:
        raise RuntimeError("没有可用的动画时间范围，请检查 route_edges.csv。")

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
            pos, state_label, active_event = get_agv_state_at_time(
                positions=positions,
                edge_geometry=edge_geometry,
                events=agv_events.get(agv.id, []),
                agv_start_node=agv.start_node,
                t=t,
            )
            active_states.append((agv, pos, state_label, active_event))

        # 先画所有 AGV 的未来规划路径，再画 AGV 图标，避免高亮线盖住小车。
        for _, _, _, active_event in active_states:
            draw_remaining_route(
                ax=ax,
                edge_geometry=edge_geometry,
                route_edges=route_edges,
                active_event=active_event,
                t=t,
            )

        for agv_index, (agv, pos, state_label, _) in enumerate(active_states):
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

        ax.set_title(f"AGV Dispatch Animation | {run_id} | t = {t:.1f} s")
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

    output_file = Path(args.output) if args.output else run_dir / "figures" / "animation.gif"
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_file, writer="pillow", fps=max(1, int(1000 / args.interval)))
    plt.close(fig)

    print(f"动画已保存：{output_file}")
    print("动画数据源：route_edges.csv + route_waits.csv")


if __name__ == "__main__":
    main()
