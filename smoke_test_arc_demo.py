from pathlib import Path
import math

import matplotlib.pyplot as plt
import networkx as nx
import simpy


# ============================================================
# 1. 几何工具函数：直线、圆弧、长度计算
# ============================================================

def make_line_points(start, end):
    """
    生成直线路径点。
    这里直线只需要起点和终点两个点。
    """
    return [start, end]


def make_arc_points(center, radius, start_deg, end_deg, num=30):
    """
    根据圆心、半径、起止角度，自动生成圆弧中间点。

    center: 圆心坐标，例如 (4, 2)
    radius: 半径，例如 2
    start_deg: 起始角度，单位：度
    end_deg: 结束角度，单位：度
    num: 采样点数量，越大越圆滑
    """
    cx, cy = center

    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    points = []

    for i in range(num):
        if num == 1:
            t = 0
        else:
            t = i / (num - 1)

        angle = start_rad + t * (end_rad - start_rad)

        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        points.append((x, y))

    return points


def polyline_length(points):
    """
    计算一串点组成的折线/圆弧近似路径长度。
    圆弧已经被采样成很多点，所以也可以用这个函数算长度。
    """
    total = 0.0

    for p1, p2 in zip(points[:-1], points[1:]):
        x1, y1 = p1
        x2, y2 = p2
        total += math.hypot(x2 - x1, y2 - y1)

    return total


def point_at_fraction(points, fraction=0.5):
    """
    按照路径真实长度比例取点。
    fraction=0.5 表示取这条边长度一半的位置。

    不做偏移，这样后面节点多了也不用维护 offset。
    """
    if not points:
        raise ValueError("points 不能为空")

    if len(points) == 1:
        return points[0]

    total_len = polyline_length(points)

    if total_len == 0:
        return points[0]

    target_len = total_len * fraction
    acc_len = 0.0

    for p1, p2 in zip(points[:-1], points[1:]):
        x1, y1 = p1
        x2, y2 = p2

        seg_len = math.hypot(x2 - x1, y2 - y1)

        if seg_len == 0:
            continue

        if acc_len + seg_len >= target_len:
            remain = target_len - acc_len
            ratio = remain / seg_len

            x = x1 + ratio * (x2 - x1)
            y = y1 + ratio * (y2 - y1)

            return (x, y)

        acc_len += seg_len

    return points[-1]


def is_close_point(p1, p2, eps=1e-6):
    """
    判断两个点是否足够接近。
    用来检查圆弧起点/终点是否和节点坐标对得上。
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < eps


# ============================================================
# 2. 根据 positions 和 edge_defs 自动建立图和显示几何
# ============================================================

def build_graph_from_defs(positions, edge_defs):
    """
    根据边定义自动建立 NetworkX 图。

    返回：
    graph: 给仿真和最短路径用
    edge_geometry: 给画图用

    注意：
    edge_geometry 里的圆弧中间点不是 NetworkX 节点，
    它们只参与显示，不参与调度。
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(positions.keys())

    edge_geometry = {}

    for edge in edge_defs:
        u = edge["u"]
        v = edge["v"]
        kind = edge["kind"]

        if kind == "line":
            points = make_line_points(
                positions[u],
                positions[v],
            )

        elif kind == "arc":
            points = make_arc_points(
                center=edge["center"],
                radius=edge["radius"],
                start_deg=edge["start_deg"],
                end_deg=edge["end_deg"],
                num=edge.get("num", 30),
            )

            # 检查圆弧起点、终点是否刚好接到两个节点
            if not is_close_point(points[0], positions[u]):
                raise ValueError(
                    f"{u}->{v} 的圆弧起点 {points[0]} 和节点 {u} 坐标 {positions[u]} 不一致"
                )

            if not is_close_point(points[-1], positions[v]):
                raise ValueError(
                    f"{u}->{v} 的圆弧终点 {points[-1]} 和节点 {v} 坐标 {positions[v]} 不一致"
                )

        else:
            raise ValueError(f"未知边类型：{kind}")

        distance = polyline_length(points)

        graph.add_edge(
            u,
            v,
            weight=round(distance, 3),
        )

        edge_geometry[(u, v)] = points

    return graph, edge_geometry


# ============================================================
# 3. 自定义画图函数：支持直线边和 90 度圆弧边
# ============================================================

def draw_network_with_geometry(graph, positions, edge_geometry, output_path):
    """
    画 AGV 路网。

    - 节点：用 positions 画
    - 边：用 edge_geometry 画
    - 边权标签：自动放在路径长度 50% 的位置，不做偏移
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 画边
    for u, v, data in graph.edges(data=True):
        points = edge_geometry[(u, v)]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(xs, ys, linewidth=2.5, color="black")

        # 画箭头
        # 圆弧有很多点，直线只有两个点，都可以这样处理
        if len(points) >= 2:
            ax.annotate(
                "",
                xy=points[-1],
                xytext=points[-2],
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.5,
                    color="black",
                ),
            )

        # 画边权标签，不加偏移
        label_x, label_y = point_at_fraction(points, fraction=0.5)

        ax.text(
            label_x,
            label_y,
            f'{data["weight"]:.2f}',
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )

    # 画节点
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        node_size=1600,
        node_color="lightblue",
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos=positions,
        font_size=12,
        font_weight="bold",
        ax=ax,
    )

    # 自动计算边界，避免裁剪
    all_points = []

    for points in edge_geometry.values():
        all_points.extend(points)

    all_points.extend(positions.values())

    xs_all = [p[0] for p in all_points]
    ys_all = [p[1] for p in all_points]

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)

    pad_x = max((x_max - x_min) * 0.08, 0.8)
    pad_y = max((y_max - y_min) * 0.25, 0.8)

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.title("AGV Road Network Smoke Test with Line and 90-degree Arc")
    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close()


# ============================================================
# 4. 定义节点和边
# ============================================================

# 节点坐标
# 后面你有更多点，就继续往这里加
positions = {
    "S": (0, 0),
    "A": (4, 0),
    "B": (6, 2),
    "C": (10, 2),
    "D": (12, 0),
    "E": (16, 0),
}

# 边定义
# kind="line" 表示直线
# kind="arc" 表示四分之一圆弧
#
# 注意：
# 圆弧的起点和终点必须刚好对应 u、v 的坐标。
# 例如 A=(4,0), B=(6,2)
# 圆心 center=(4,2), 半径=2
# 270° 对应 (4,0)，360° 对应 (6,2)
edge_defs = [
    {
        "u": "S",
        "v": "A",
        "kind": "line",
    },
    {
        "u": "A",
        "v": "B",
        "kind": "arc",
        "center": (4, 2),
        "radius": 2,
        "start_deg": 270,
        "end_deg": 360,
        "num": 30,
    },
    {
        "u": "B",
        "v": "C",
        "kind": "line",
    },
    {
        "u": "C",
        "v": "D",
        "kind": "arc",
        "center": (12, 2),
        "radius": 2,
        "start_deg": 180,
        "end_deg": 270,
        "num": 30,
    },
    {
        "u": "D",
        "v": "E",
        "kind": "line",
    },
]


# ============================================================
# 5. 建立图，计算最短路径
# ============================================================

graph, edge_geometry = build_graph_from_defs(
    positions=positions,
    edge_defs=edge_defs,
)

start_node = "S"
target_node = "E"

path = nx.shortest_path(
    graph,
    source=start_node,
    target=target_node,
    weight="weight",
)

distance = nx.shortest_path_length(
    graph,
    source=start_node,
    target=target_node,
    weight="weight",
)

print("最短路径:", path)
print("路径距离:", distance)


# ============================================================
# 6. 使用 SimPy 模拟 AGV 行驶
# ============================================================

def agv_process(env: simpy.Environment, agv_id: str, speed: float):
    print(f"[{env.now:>5.1f} s] {agv_id} 从 {start_node} 出发")

    travel_time = distance / speed
    yield env.timeout(travel_time)

    print(f"[{env.now:>5.1f} s] {agv_id} 到达 {target_node}")


env = simpy.Environment()
env.process(agv_process(env, agv_id="AGV01", speed=1.0))
env.run()


# ============================================================
# 7. 绘制路网并保存图片
# ============================================================

Path("outputs").mkdir(exist_ok=True)

output_path = "outputs/smoke_network_arc_demo.png"

draw_network_with_geometry(
    graph=graph,
    positions=positions,
    edge_geometry=edge_geometry,
    output_path=output_path,
)

print(f"路网图片已保存: {output_path}")
print("环境测试成功")