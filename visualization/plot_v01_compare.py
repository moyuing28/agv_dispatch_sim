from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        df["run_id"] = run_dir.name
    return df


def add_value_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3g", padding=3, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="要对比的实验目录，例如 outputs/runs/v01_fcfs_time_stepwise outputs/runs/v01_nearest_time_stepwise。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "figures"),
        help="跨实验对比图输出目录。默认 outputs/figures。",
    )
    args = parser.parse_args()

    run_dirs = [resolve_path(item) for item in args.run_dirs]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.concat(
        [read_summary(run_dir) for run_dir in run_dirs],
        ignore_index=True,
    )

    label_col = "run_id"
    compare_csv = output_dir / "compare_summary.csv"
    summary_df.to_csv(compare_csv, index=False)

    print("========== Comparison Summary ==========")
    print(summary_df.to_string(index=False))

    candidate_metrics = [
        ("makespan", "Makespan / s"),
        ("avg_wait_time", "Average Waiting Time / s"),
        ("avg_flow_time", "Average Flow Time / s"),
        ("total_distance", "Total Distance"),
        ("empty_distance", "Empty Distance"),
        ("empty_rate", "Empty Rate"),
        ("traffic_wait_total", "Traffic Wait Total / s"),
        ("internal_wait_total", "Internal Wait Total / s"),
        ("traffic_conflict_count", "Traffic Conflict Count"),
    ]
    metrics = [(m, title) for m, title in candidate_metrics if m in summary_df.columns]

    if not metrics:
        raise RuntimeError("No comparable metrics found in summary files.")

    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(summary_df[label_col], summary_df[metric])
        ax.set_title(title)
        ax.set_xlabel("Run")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=30)
        add_value_labels(ax)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Dispatch Experiment Comparison", fontsize=14)
    fig.tight_layout()

    output_file = output_dir / "compare_kpi.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"\nComparison summary saved to: {compare_csv}")
    print(f"KPI comparison figure saved to: {output_file}")


if __name__ == "__main__":
    main()
