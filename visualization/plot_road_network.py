from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.road_network import build_demo_network, point_at_fraction


# 和动画脚本保持一致：道路统一为蓝色。
ROAD_COLOR = "#1f77b4"
ROAD_LINE_WIDTH = 1.8
NODE_COLOR = "#f2f2f2"


def resolve_run_dir(run_dir_arg: str | None) -> Path | None:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        return run_dir

    latest_path = PROJECT_ROOT / "outputs" / "latest_run.json"
    if latest_path.exists():
        with latest_path.open("r", encoding="utf-8") as f:
            latest = json.load(f)
        return Path(latest["run_dir"])

    return None


def read_run_config(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {"run_id": run_dir.name}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_network_with_geometry(
    graph,
    positions,
    edge_geometry,
    output_path,
    show_node_labels=True,
    show_edge_lengths=True,
    title="AGV Road Network",
):
    fig, ax = plt.subplots(figsize=(14, 9))

    for u, v, data in graph.edges(data=True):
        # 双向边会重复画两次，为了画面不太黑，只画字典序较小的一边。
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
                f'{data["length"]:.2f}',
                fontsize=6,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                zorder=3,
            )

    node_collection = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_size=100,
        node_color=NODE_COLOR,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )
    node_collection.set_zorder(4)

    if show_node_labels:
        label_artists = nx.draw_networkx_labels(
            graph,
            pos=positions,
            font_size=6,
            font_weight="bold",
            ax=ax,
        )
        for label in label_artists.values():
            label.set_zorder(5)

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

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help="实验输出目录。若不指定，优先读取 outputs/latest_run.json。",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="地图配置模块，例如 configs.demo_map。若不指定，优先从 run_dir/config.json 读取。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出图片路径。若不指定，有 run_dir 时保存到 run_dir/figures/road_network.png，否则保存到 outputs/figures/road_network.png。",
    )
    parser.add_argument(
        "--hide-node-labels",
        action="store_true",
        help="隐藏节点编号",
    )
    parser.add_argument(
        "--hide-edge-lengths",
        action="store_true",
        help="隐藏边的真实长度标注",
    )

    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    config = read_run_config(run_dir)
    map_module = args.map or config.get("map") or config.get("args", {}).get("map") or "configs.demo_map"
    run_id = config.get("run_id", run_dir.name if run_dir else map_module)

    graph, positions, edge_geometry = build_demo_network(
        return_geometry=True,
        map_module=map_module,
    )

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    elif run_dir is not None:
        output_path = run_dir / "figures" / "road_network.png"
    else:
        output_path = PROJECT_ROOT / "outputs" / "figures" / "road_network.png"

    draw_network_with_geometry(
        graph=graph,
        positions=positions,
        edge_geometry=edge_geometry,
        output_path=output_path,
        show_node_labels=not args.hide_node_labels,
        show_edge_lengths=not args.hide_edge_lengths,
        title=f"AGV Road Network | {run_id}",
    )

    print(f"路网图片已保存：{output_path}")
    print(f"节点数量：{graph.number_of_nodes()}")
    print(f"有向边数量：{graph.number_of_edges()}")


if __name__ == "__main__":
    main()
