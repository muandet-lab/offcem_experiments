"""Plot K-means granularity schedules alongside legacy OPE baselines."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/Users/cispa/Documents/OffCEM/k-means-analysis"),
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=Path(
            "/Users/cispa/Documents/OffCEM/"
            "sample_size_stress_results/sample_size_stress"
        ),
        help="Directory containing the original sample-size aggregate CSV.",
    )
    parser.add_argument("--out-file", type=Path, default=None)
    return parser.parse_args()


def prepare_comparison(aggregates: pd.DataFrame) -> pd.DataFrame:
    required = {"n", "kmeans_k", "rel_mse"}
    if missing := required - set(aggregates.columns):
        raise ValueError(f"aggregate CSV is missing columns: {sorted(missing)}")
    baseline = aggregates[aggregates["kmeans_k"] == 50][["n", "rel_mse"]].rename(
        columns={"rel_mse": "fixed_k50_rel_mse"}
    )
    if baseline["n"].nunique() != aggregates["n"].nunique():
        raise ValueError("every sample size must include the fixed K=50 baseline")
    best = aggregates.loc[aggregates.groupby("n")["rel_mse"].idxmin()][
        ["n", "kmeans_k", "rel_mse"]
    ].rename(columns={"kmeans_k": "best_k", "rel_mse": "best_rel_mse"})
    comparison = baseline.merge(best, on="n", validate="one_to_one").sort_values("n")
    comparison["relative_improvement"] = 1.0 - (
        comparison["best_rel_mse"] / comparison["fixed_k50_rel_mse"]
    )
    return comparison


def load_legacy_baselines(results_dir: Path) -> pd.DataFrame:
    aggregates = pd.read_csv(results_dir / "sample_size_stress_aggregate.csv")
    requested = ["OffCEM matched", "DR", "DM"]
    baselines = aggregates[aggregates["estimator"].isin(requested)].copy()
    missing = set(requested) - set(baselines["estimator"])
    if missing:
        raise ValueError(f"legacy aggregate CSV is missing: {sorted(missing)}")
    return baselines


def select_kmeans_schedules(aggregates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build interpretable low/middle/high K paths from each tested n grid."""
    ordered = aggregates.sort_values(["n", "kmeans_k"])
    schedules = {"Fixed K=50": [], "Smallest tested K": [], "Middle tested K": [], "Largest tested K": []}
    for _, rows in ordered.groupby("n", sort=True):
        rows = rows.reset_index(drop=True)
        fixed = rows[rows["kmeans_k"] == 50]
        if len(fixed) != 1:
            raise ValueError("every sample size must include exactly one K=50 row")
        schedules["Fixed K=50"].append(fixed.iloc[0])
        schedules["Smallest tested K"].append(rows.iloc[0])
        schedules["Middle tested K"].append(rows.iloc[len(rows) // 2])
        schedules["Largest tested K"].append(rows.iloc[-1])
    return {name: pd.DataFrame(rows) for name, rows in schedules.items()}


def make_plot(
    kmeans_aggregates: pd.DataFrame,
    legacy_baselines: pd.DataFrame,
    output: Path,
) -> None:
    figure, error_axis = plt.subplots(figsize=(9, 5.5))
    legacy_styles = {
        "OffCEM matched": ("#1f77b4", "OffCEM matched (original)"),
        "DR": ("#d62728", "DR (original)"),
        "DM": ("#9467bd", "DM (original)"),
    }
    for estimator, (color, label) in legacy_styles.items():
        rows = legacy_baselines[legacy_baselines["estimator"] == estimator].sort_values("n")
        error_axis.plot(
            rows["n"],
            rows["rel_mse"],
            marker="o",
            linewidth=1.6,
            color=color,
            label=label,
        )

    # K changes with n, so every (n, K) result is retained as a background point.
    # Lines connect rank-defined K schedules, not identical K values across n.
    kmeans_aggregates = kmeans_aggregates.sort_values(["n", "kmeans_k"])
    for n, rows in kmeans_aggregates.groupby("n", sort=True):
        rows = rows.sort_values("kmeans_k").reset_index(drop=True)
        offsets = np.linspace(0.88, 1.12, len(rows))
        for row, offset in zip(rows.itertuples(index=False), offsets):
            error_axis.scatter(
                n * offset,
                row.rel_mse,
                s=30,
                color="#9a9a9a",
                edgecolor="white",
                linewidth=0.7,
                alpha=0.55,
                zorder=4,
                label="Other tested K-means cases" if n == kmeans_aggregates["n"].min() and row.kmeans_k == rows.iloc[0]["kmeans_k"] else None,
            )

    schedule_styles = {
        "Fixed K=50": ("#4c4c4c", "o"),
        "Smallest tested K": ("#1f77b4", "s"),
        "Middle tested K": ("#ff7f0e", "D"),
        "Largest tested K": ("#2ca02c", "^"),
    }
    for name, rows in select_kmeans_schedules(kmeans_aggregates).items():
        color, marker = schedule_styles[name]
        error_axis.plot(
            rows["n"],
            rows["rel_mse"],
            marker=marker,
            markersize=6,
            linewidth=1.8,
            color=color,
            label=name,
            zorder=5,
        )
        for row in rows.itertuples(index=False):
            error_axis.annotate(
                f"{int(row.kmeans_k)}",
                (row.n, row.rel_mse),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=color,
            )
    error_axis.set_xscale("log")
    error_axis.set_xlabel("logged interactions n")
    error_axis.set_ylabel("Relative MSE")
    error_axis.grid(True, alpha=0.3, which="both")
    error_axis.legend(fontsize=6.5, loc="lower left", ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    aggregate_path = args.results_dir / "kmeans_granularity_aggregate.csv"
    output = args.out_file or args.results_dir / "adaptive_k_vs_fixed_k50.png"
    kmeans_aggregates = pd.read_csv(aggregate_path)
    prepare_comparison(kmeans_aggregates)  # Validate the intended fixed-K baseline.
    legacy_baselines = load_legacy_baselines(args.baseline_results_dir)
    make_plot(kmeans_aggregates, legacy_baselines, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
