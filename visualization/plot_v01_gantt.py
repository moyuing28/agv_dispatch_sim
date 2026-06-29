from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        return run_dir

    latest_path = PROJECT_ROOT / "outputs" / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing file: {latest_path}\n"
            "Please run experiments/run_v01.py first, or pass --run-dir."
        )
    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)
    return Path(latest["run_dir"])


def read_run_id(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return run_dir.name
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("run_id", run_dir.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help="实验输出目录，例如 outputs/runs/v01_nearest_time_stepwise。默认读取 outputs/latest_run.json。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="甘特图输出路径。若不指定，保存到 run_dir/figures/gantt.png。",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    run_id = read_run_id(run_dir)
    travel_file = run_dir / "travels.csv"

    if not travel_file.exists():
        raise FileNotFoundError(
            f"Missing file: {travel_file}\n"
            "Please run experiments/run_v01.py first."
        )

    df = pd.read_csv(travel_file)

    if df.empty:
        raise RuntimeError("Travel record file is empty.")

    df = df.sort_values(["agv_id", "start_time", "end_time"])

    agv_ids = sorted(df["agv_id"].unique())
    y_map = {agv_id: i for i, agv_id in enumerate(agv_ids)}

    fig, ax = plt.subplots(figsize=(12, 4.8))

    for _, row in df.iterrows():
        agv_id = row["agv_id"]
        task_id = row["task_id"]
        mode = row["mode"]
        start = float(row["start_time"])
        end = float(row["end_time"])
        duration = end - start

        y = y_map[agv_id]

        if duration <= 0:
            ax.scatter(start, y, marker="o", s=40)
            ax.text(
                start,
                y + 0.08,
                f"{task_id}-{mode}",
                fontsize=7,
                ha="center",
                va="bottom",
            )
            continue

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.36,
        )

        ax.text(
            start + duration / 2,
            y,
            f"{task_id}-{mode}",
            fontsize=7,
            ha="center",
            va="center",
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Simulation Time / s")
    ax.set_ylabel("AGV")
    ax.set_title(f"AGV Task Execution Gantt Chart | {run_id}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_file = Path(args.output) if args.output else run_dir / "figures" / "gantt.png"
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"Gantt chart saved to: {output_file}")


if __name__ == "__main__":
    main()
