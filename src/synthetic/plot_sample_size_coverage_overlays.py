"""Overlay empirical evaluation coverage on sample-size error curves.

Population support and policy overlap are fixed within the experiment. The
translucent bars therefore show the quantity that actually changes with each
evaluation prefix: target-policy mass on actions observed in that prefix.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_sample_size_full_scale import COLORS
from plot_sample_size_full_scale import DEFAULT_ROOT
from plot_sample_size_full_scale import ORDER
from plot_sample_size_full_scale import _population_overlap_metrics
from plot_sample_size_full_scale import load_and_merge
from plot_sample_size_full_scale import standard_error


MATCHED_ESTIMATOR = "OffCEM matched"
COVERAGE_COLUMN = "target_mass_on_observed_actions"
COVERAGE_COLOR = "#374151"
COVERAGE_ALPHA = 0.30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=DEFAULT_ROOT / "sample_size_hierarchical_pilot",
    )
    parser.add_argument(
        "--extended-dir",
        type=Path,
        default=DEFAULT_ROOT / "larger_case",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT / "sample_size_coverage_overlays",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def load_end_to_end(pilot_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(pilot_dir / "sample_size_stress_tidy.csv")
    return rows[rows["arm"] == "end_to_end"].copy()


def _coverage_by_n(rows: pd.DataFrame) -> dict:
    diagnostics = rows[rows["estimator"] == MATCHED_ESTIMATOR]
    if diagnostics.empty or diagnostics[COVERAGE_COLUMN].isna().any():
        raise ValueError("matched OffCEM coverage diagnostics are missing")
    return {
        int(n): float(n_rows[COVERAGE_COLUMN].mean())
        for n, n_rows in diagnostics.groupby("n")
    }


def _fixed_policy_note(rows: pd.DataFrame) -> str:
    metrics = [_population_overlap_metrics(int(seed)) for seed in rows["world_seed"].unique()]
    action_ess = np.mean([item["action_ess_fraction"] for item in metrics])
    cluster_ess = np.mean([item["cluster_ess_fraction"] for item in metrics])
    return (
        "Fixed population: action and cluster support = 1.00; "
        f"ESS/n action = {action_ess:.1e}, matched cluster = {cluster_ess:.1e}"
    )


def make_overlay(
    rows: pd.DataFrame,
    metric: str,
    ylabel: str,
    output: Path,
    n_bootstrap: int,
    log_y: bool,
) -> Path:
    coverage = _coverage_by_n(rows)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    coverage_axis = axis.twinx()
    x_coverage = np.asarray(sorted(coverage))
    coverage_axis.bar(
        x_coverage,
        [coverage[n] for n in x_coverage],
        width=x_coverage * 0.34,
        color=COVERAGE_COLOR,
        alpha=COVERAGE_ALPHA,
        linewidth=0,
        label="Target-policy mass on observed actions",
        zorder=0,
    )
    coverage_axis.set_ylim(0.0, 1.02)
    coverage_axis.set_ylabel("target-policy mass observed")
    coverage_axis.set_zorder(0)
    axis.set_zorder(1)
    axis.patch.set_visible(False)

    available = set(rows["estimator"])
    estimators = [estimator for estimator in ORDER if estimator in available]
    for estimator_index, estimator in enumerate(estimators):
        selected = rows[rows["estimator"] == estimator]
        x_values, points, standard_errors = [], [], []
        for n, n_rows in selected.groupby("n"):
            point, error = standard_error(n_rows, metric)
            x_values.append(n)
            points.append(point)
            standard_errors.append(error)
        order = np.argsort(x_values)
        x_values = np.asarray(x_values)[order]
        points = np.asarray(points)[order]
        standard_errors = np.asarray(standard_errors)[order]
        axis.errorbar(
            x_values,
            points,
            yerr=standard_errors,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=3,
            elinewidth=1.0,
            color=COLORS[estimator],
            label=estimator,
        )

    axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
    axis.set_xlabel("logged evaluation interactions n")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3, which="both")
    line_handles, line_labels = axis.get_legend_handles_labels()
    line_handles.append(
        Patch(
            facecolor=COVERAGE_COLOR,
            alpha=COVERAGE_ALPHA,
            label="Target-policy mass on observed actions",
        )
    )
    line_labels.append("Target-policy mass on observed actions")
    axis.legend(line_handles, line_labels, fontsize=7, loc="lower left")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    end_to_end = load_end_to_end(args.pilot_dir)
    frozen = load_and_merge(args.pilot_dir, args.extended_dir)
    specifications = [
        ("end_to_end", end_to_end, False),
        ("frozen", frozen, True),
    ]
    for arm, rows, log_y in specifications:
        for metric, ylabel, stem in [
            ("rel_mse", "Relative MSE", "rel_mse"),
            ("bias2", "Bias^2", "bias2"),
        ]:
            output = make_overlay(
                rows,
                metric,
                ylabel,
                args.out_dir / f"{arm}_{stem}_with_coverage.png",
                args.n_bootstrap,
                log_y,
            )
            print(f"wrote {output}")


if __name__ == "__main__":
    main()
