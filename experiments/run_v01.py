from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policies.fcfs_policy import FCFSPolicy
from policies.nearest_agv_policy import NearestAGVPolicy
from simulator.engine import SimulationEngine
from simulator.models import AGV, Task
from simulator.road_network import build_demo_network


def build_agvs():
    return [
        AGV(id="AGV01", start_node="1"),
        AGV(id="AGV02", start_node="36"),
        AGV(id="AGV03", start_node="92"),
    ]


def build_tasks():
    return [
        Task(id="T01", release_time=0, pickup_node="1", dropoff_node="64"),
        Task(id="T02", release_time=0, pickup_node="12", dropoff_node="112"),
        Task(id="T03", release_time=10, pickup_node="28", dropoff_node="77"),
        Task(id="T04", release_time=20, pickup_node="47", dropoff_node="5"),
        Task(id="T05", release_time=35, pickup_node="69", dropoff_node="31"),
        Task(id="T06", release_time=50, pickup_node="83", dropoff_node="124"),
        Task(id="T07", release_time=70, pickup_node="101", dropoff_node="18"),
        Task(id="T08", release_time=90, pickup_node="116", dropoff_node="55"),
        Task(id="T09", release_time=110, pickup_node="73", dropoff_node="8"),
        Task(id="T10", release_time=130, pickup_node="126", dropoff_node="40"),
    ]


def get_policy(name: str):
    if name == "fcfs":
        return FCFSPolicy()

    if name == "nearest":
        return NearestAGVPolicy()

    raise ValueError(f"未知策略：{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["fcfs", "nearest"],
        default="fcfs",
        help="选择调度策略",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印详细事件日志",
    )
    parser.add_argument(
        "--map",
        default="configs.demo_map",
        help="地图配置模块，例如 configs.demo_map",
    )

    args = parser.parse_args()

    graph, _ = build_demo_network(map_module=args.map)
    agvs = build_agvs()
    tasks = build_tasks()
    policy = get_policy(args.policy)

    engine = SimulationEngine(
        graph=graph,
        agvs=agvs,
        tasks=tasks,
        policy=policy,
        verbose=not args.quiet,
    )

    summary = engine.run()

    print("\n========== 仿真汇总 ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    pd.DataFrame([summary]).to_csv(
        output_dir / f"v01_summary_{policy.name}.csv",
        index=False,
    )

    pd.DataFrame([task.__dict__ for task in engine.completed_tasks]).to_csv(
        output_dir / f"v01_tasks_{policy.name}.csv",
        index=False,
    )

    pd.DataFrame(engine.event_log).to_csv(
        output_dir / f"v01_events_{policy.name}.csv",
        index=False,
    )

    pd.DataFrame(engine.travel_records).to_csv(
        output_dir / f"v01_travels_{policy.name}.csv",
        index=False,
    )

    print(f"\n结果已保存到 outputs/v01_summary_{policy.name}.csv")
    print(f"任务明细已保存到 outputs/v01_tasks_{policy.name}.csv")
    print(f"事件日志已保存到 outputs/v01_events_{policy.name}.csv")
    print(f"轨迹记录已保存到 outputs/v01_travels_{policy.name}.csv")


if __name__ == "__main__":
    main()
