from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["fcfs", "nearest"],
        default="nearest",
        help="Dispatch policy to plot",
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "outputs"
    travel_file = output_dir / f"v01_travels_{args.policy}.csv"

    if not travel_file.exists():
        raise FileNotFoundError(
            f"Missing file: {travel_file}\n"
            f"Please run: python experiments/run_v01.py --policy {args.policy} --quiet"
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
    ax.set_title(f"AGV Task Execution Gantt Chart | Policy: {args.policy}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_file = output_dir / f"v01_gantt_{args.policy}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"Gantt chart saved to: {output_file}")


if __name__ == "__main__":
    main()
