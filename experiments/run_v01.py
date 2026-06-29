from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policies.fcfs_policy import FCFSPolicy
from policies.nearest_agv_policy import NearestAGVPolicy
from simulator.cost_models import get_cost_model
from simulator.engine import SimulationEngine
from simulator.models import AGV, Task
from simulator.road_network import build_demo_network
from simulator.route_planner import RoutePlanner
from simulator.traffic_manager import TrafficManager


OUTPUT_LEVELS = {"basic": 1, "normal": 2, "debug": 3}


def build_agvs():
    return [
        AGV(id="AGV01", start_node="1"),
        AGV(id="AGV02", start_node="8"),
        # AGV(id="AGV03", start_node="92"),
    ]


def build_tasks():
    return [
        Task(id="T01", release_time=0, pickup_node="2", dropoff_node="7"),
        Task(id="T02", release_time=0, pickup_node="5", dropoff_node="1"),
        # Task(id="T03", release_time=10, pickup_node="28", dropoff_node="77"),
        # Task(id="T04", release_time=20, pickup_node="47", dropoff_node="5"),
        # Task(id="T05", release_time=35, pickup_node="69", dropoff_node="31"),
        # Task(id="T06", release_time=50, pickup_node="83", dropoff_node="120"),
        # Task(id="T07", release_time=70, pickup_node="101", dropoff_node="18"),
        # Task(id="T08", release_time=90, pickup_node="116", dropoff_node="55"),
        # Task(id="T09", release_time=110, pickup_node="73", dropoff_node="8"),
        # Task(id="T10", release_time=130, pickup_node="111", dropoff_node="40"),
    ]


def get_policy(name: str):
    if name == "fcfs":
        return FCFSPolicy()

    if name == "nearest":
        return NearestAGVPolicy()

    raise ValueError(f"未知策略：{name}")


def safe_name(value: str) -> str:
    """把用户输入转成安全的文件夹名。"""
    value = str(value).strip()
    value = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", value)
    value = value.strip("._-")
    return value or "run"


def build_run_id(version: str, policy_name: str, cost_model_name: str, traffic_mode: str, run_name: str | None) -> str:
    if run_name:
        return safe_name(run_name)
    return safe_name(f"{version}_{policy_name}_{cost_model_name}_{traffic_mode}")


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def maybe_write(outputs: dict[str, Path], label: str, df: pd.DataFrame, path: Path) -> None:
    write_csv(df, path)
    outputs[label] = path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["fcfs", "nearest"],
        default="fcfs",
        help="选择调度策略",
    )
    parser.add_argument(
        "--cost-model",
        choices=["distance", "time"],
        default="distance",
        help="选择路径规划代价模型：distance=距离最短，time=时间最短",
    )
    parser.add_argument(
        "--traffic-mode",
        choices=["none", "reservation", "stepwise", "reroute"],
        default="stepwise",
        help="路径冲突处理模式：none=不处理冲突，reservation=整条路径等待，stepwise=固定路径逐边预约/中途等待，reroute=候选路径中选择实际到达最早",
    )
    parser.add_argument(
        "--traffic-safety-time",
        type=float,
        default=0.5,
        help="预约表安全缓冲时间，单位秒。默认 0.5。",
    )
    parser.add_argument(
        "--node-hold-time",
        type=float,
        default=0.5,
        help="节点占用时间，单位秒。默认 0.5。",
    )
    parser.add_argument(
        "--line-speed-empty",
        type=float,
        default=None,
        help="time 模型：空载直线路段速度。默认使用 AGV.speed_empty。",
    )
    parser.add_argument(
        "--line-speed-loaded",
        type=float,
        default=None,
        help="time 模型：载货直线路段速度。默认使用 AGV.speed_loaded。",
    )
    parser.add_argument(
        "--arc-speed-empty",
        type=float,
        default=None,
        help="time 模型：空载弯道路段速度。若不指定，则使用直线速度 * arc-speed-ratio。",
    )
    parser.add_argument(
        "--arc-speed-loaded",
        type=float,
        default=None,
        help="time 模型：载货弯道路段速度。若不指定，则使用直线速度 * arc-speed-ratio。",
    )
    parser.add_argument(
        "--arc-speed-ratio",
        type=float,
        default=0.5,
        help="time 模型：弯道速度/直线速度比例。默认 0.5，即弯道速度为直线一半。",
    )
    parser.add_argument(
        "--arc-delay",
        type=float,
        default=0.0,
        help="time 模型：每经过一条弯道边额外增加的固定时间，单位秒。默认 0。",
    )

    parser.add_argument(
        "--candidate-paths",
        type=int,
        default=5,
        help="reroute 模式下评估的候选路径数量。默认 5。",
    )
    parser.add_argument(
        "--output-level",
        choices=["basic", "normal", "debug"],
        default="normal",
        help="输出详细程度：basic=最少文件，normal=论文分析常用，debug=完整调试文件。默认 normal。",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="自定义本次运行文件夹名。若不指定，自动使用 v01_policy_cost_traffic。",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "runs"),
        help="实验结果根目录。默认 outputs/runs。",
    )
    parser.add_argument(
        "--version",
        default="v01",
        help="实验版本号，用于自动生成 run_id。默认 v01。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印详细事件日志",
    )
    parser.add_argument(
        "--map",
        default="configs.first_map",
        help="地图配置模块，例如 configs.demo_map",
    )

    args = parser.parse_args()

    if args.arc_speed_ratio <= 0:
        raise ValueError("--arc-speed-ratio 必须大于 0")
    if args.arc_delay < 0:
        raise ValueError("--arc-delay 不能小于 0")
    if args.traffic_safety_time < 0:
        raise ValueError("--traffic-safety-time 不能小于 0")
    if args.node_hold_time < 0:
        raise ValueError("--node-hold-time 不能小于 0")
    if args.candidate_paths <= 0:
        raise ValueError("--candidate-paths 必须大于 0")

    graph, _ = build_demo_network(map_module=args.map)
    agvs = build_agvs()
    tasks = build_tasks()
    policy = get_policy(args.policy)
    cost_model = get_cost_model(
        args.cost_model,
        line_speed_empty=args.line_speed_empty,
        line_speed_loaded=args.line_speed_loaded,
        arc_speed_empty=args.arc_speed_empty,
        arc_speed_loaded=args.arc_speed_loaded,
        arc_speed_ratio=args.arc_speed_ratio,
        arc_delay=args.arc_delay,
    )
    route_planner = RoutePlanner(cost_model=cost_model)

    traffic_manager = None
    if args.traffic_mode in {"reservation", "stepwise", "reroute"}:
        if args.traffic_mode == "reservation":
            traffic_strategy = "whole_route"
        elif args.traffic_mode == "stepwise":
            traffic_strategy = "stepwise"
        else:
            traffic_strategy = "reroute"

        traffic_manager = TrafficManager(
            safety_time=args.traffic_safety_time,
            node_hold_time=args.node_hold_time,
            strategy=traffic_strategy,
            candidate_path_count=args.candidate_paths,
        )

    engine = SimulationEngine(
        graph=graph,
        agvs=agvs,
        tasks=tasks,
        policy=policy,
        verbose=not args.quiet,
        route_planner=route_planner,
        traffic_manager=traffic_manager,
    )

    summary = engine.run()
    summary.update(cost_model.parameters())

    run_id = build_run_id(
        version=args.version,
        policy_name=policy.name,
        cost_model_name=cost_model.name,
        traffic_mode=args.traffic_mode,
        run_name=args.run_name,
    )
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_dir": str(run_dir),
        "figures_dir": str(figures_dir),
        "args": vars(args),
        "policy": policy.name,
        "cost_model": cost_model.name,
        "traffic_mode": args.traffic_mode,
        "map": args.map,
        "cost_model_parameters": cost_model.parameters(),
    }

    summary.update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "output_level": args.output_level,
        }
    )

    summary_df = pd.DataFrame([summary])
    tasks_df = pd.DataFrame([task.__dict__ for task in engine.completed_tasks])
    travels_df = pd.DataFrame(engine.travel_records)
    route_edges_df = pd.DataFrame(engine.route_edge_records)
    route_nodes_df = pd.DataFrame(engine.route_node_records)
    route_waits_df = pd.DataFrame(engine.route_wait_records)
    events_df = pd.DataFrame(engine.event_log)

    conflict_records = []
    edge_reservations = []
    node_reservations = []
    traffic_waits = []
    reroute_records = []
    if traffic_manager is not None:
        conflict_records = traffic_manager.conflict_records
        edge_reservations = traffic_manager.edge_reservations
        node_reservations = traffic_manager.node_reservations
        traffic_waits = traffic_manager.wait_records
        reroute_records = getattr(traffic_manager, "reroute_records", [])

    traffic_conflicts_df = pd.DataFrame(conflict_records)
    edge_reservations_df = pd.DataFrame(edge_reservations)
    node_reservations_df = pd.DataFrame(node_reservations)
    traffic_waits_df = pd.DataFrame(traffic_waits)
    reroute_records_df = pd.DataFrame(reroute_records)

    level = OUTPUT_LEVELS[args.output_level]
    outputs: dict[str, Path] = {}

    # 所有模式都输出：最基础的实验结果。
    maybe_write(outputs, "summary", summary_df, run_dir / "summary.csv")
    maybe_write(outputs, "tasks", tasks_df, run_dir / "tasks.csv")
    maybe_write(outputs, "travels", travels_df, run_dir / "travels.csv")
    write_json(config, run_dir / "config.json")
    outputs["config"] = run_dir / "config.json"

    # normal/debug 输出：论文分析和冲突分析常用文件。
    if level >= OUTPUT_LEVELS["normal"]:
        maybe_write(outputs, "route_edges", route_edges_df, run_dir / "route_edges.csv")
        maybe_write(outputs, "route_nodes", route_nodes_df, run_dir / "route_nodes.csv")
        maybe_write(outputs, "route_waits", route_waits_df, run_dir / "route_waits.csv")
        maybe_write(outputs, "traffic_conflicts", traffic_conflicts_df, run_dir / "traffic_conflicts.csv")
        if args.traffic_mode == "reroute":
            maybe_write(outputs, "reroute_candidates", reroute_records_df, run_dir / "reroute_candidates.csv")

    # debug 输出：详细事件和预约表明细，平时不默认保存。
    if level >= OUTPUT_LEVELS["debug"]:
        maybe_write(outputs, "events", events_df, run_dir / "events.csv")
        maybe_write(outputs, "traffic_waits", traffic_waits_df, run_dir / "traffic_waits.csv")
        maybe_write(outputs, "edge_reservations", edge_reservations_df, run_dir / "edge_reservations.csv")
        maybe_write(outputs, "node_reservations", node_reservations_df, run_dir / "node_reservations.csv")

    latest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "figures_dir": str(figures_dir),
        "policy": policy.name,
        "cost_model": cost_model.name,
        "traffic_mode": args.traffic_mode,
        "output_level": args.output_level,
        "created_at": config["created_at"],
    }
    write_json(latest, PROJECT_ROOT / "outputs" / "latest_run.json")

    print("\n========== 仿真汇总 ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n========== 输出目录 ==========")
    print(run_dir)
    print("\n========== 输出文件 ==========")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    print(f"latest_run: {PROJECT_ROOT / 'outputs' / 'latest_run.json'}")


if __name__ == "__main__":
    main()
