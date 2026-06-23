from __future__ import annotations

import argparse
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
        "--map",
        default="configs.demo_map",
        help="地图配置模块，例如 configs.demo_map",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "v01_road_network.png"),
        help="输出图片路径",
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

    graph, positions, edge_geometry = build_demo_network(
        return_geometry=True,
        map_module=args.map,
    )

    draw_network_with_geometry(
        graph=graph,
        positions=positions,
        edge_geometry=edge_geometry,
        output_path=args.output,
        show_node_labels=not args.hide_node_labels,
        show_edge_lengths=not args.hide_edge_lengths,
        title=f"AGV Road Network | {args.map}",
    )

    print(f"路网图片已保存：{args.output}")
    print(f"节点数量：{graph.number_of_nodes()}")
    print(f"有向边数量：{graph.number_of_edges()}")


if __name__ == "__main__":
    main()
