from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import simpy


# 1. 建立一个最小 AGV 路网
graph = nx.DiGraph()

graph.add_weighted_edges_from(
    [
        ("S", "A", 4.0),
        ("A", "B", 3.0),
        ("B", "D", 5.0),
        ("S", "D", 20.0),
    ]
)

start_node = "S"
target_node = "D"

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


# 2. 使用 SimPy 模拟 AGV 行驶
def agv_process(env: simpy.Environment, agv_id: str, speed: float):
    print(f"[{env.now:>5.1f} s] {agv_id} 从 {start_node} 出发")

    travel_time = distance / speed
    yield env.timeout(travel_time)

    print(f"[{env.now:>5.1f} s] {agv_id} 到达 {target_node}")


env = simpy.Environment()
env.process(agv_process(env, agv_id="AGV01", speed=1.0))
env.run()


# 3. 绘制路网并保存图片
Path("outputs").mkdir(exist_ok=True)

positions = {
    "S": (0, 0),
    "A": (1, 1),
    "B": (2, 1),
    "D": (3, 0),
}

nx.draw_networkx(
    graph,
    pos=positions,
    with_labels=True,
    node_size=1600,
    arrows=True,
)

edge_labels = nx.get_edge_attributes(graph, "weight")
nx.draw_networkx_edge_labels(
    graph,
    pos=positions,
    edge_labels=edge_labels,
)

plt.title("AGV Road Network Smoke Test")
plt.tight_layout()
plt.savefig("outputs/smoke_network.png", dpi=160)
plt.close()

print("路网图片已保存: outputs/smoke_network.png")
print("环境测试成功")