from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_summary(policy: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / f"v01_summary_{policy}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Please run: python experiments/run_v01.py --policy {policy} --quiet"
        )

    return pd.read_csv(path)


def add_value_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3g", padding=3, fontsize=8)


def main():
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    policies = ["fcfs", "nearest"]

    summary_df = pd.concat(
        [read_summary(policy) for policy in policies],
        ignore_index=True,
    )

    compare_csv = output_dir / "v01_compare_summary.csv"
    summary_df.to_csv(compare_csv, index=False)

    print("========== Comparison Summary ==========")
    print(summary_df.to_string(index=False))

    metrics = [
        ("makespan", "Makespan / s"),
        ("avg_wait_time", "Average Waiting Time / s"),
        ("avg_flow_time", "Average Flow Time / s"),
        ("total_distance", "Total Distance"),
        ("empty_distance", "Empty Distance"),
        ("empty_rate", "Empty Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(summary_df["policy"], summary_df[metric])
        ax.set_title(title)
        ax.set_xlabel("Dispatch Policy")
        ax.set_ylabel(metric)
        add_value_labels(ax)

    fig.suptitle("V0.1 Dispatch Policy Comparison", fontsize=14)
    fig.tight_layout()

    output_file = output_dir / "v01_compare_kpi.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"\nKPI comparison figure saved to: {output_file}")


if __name__ == "__main__":
    main()
