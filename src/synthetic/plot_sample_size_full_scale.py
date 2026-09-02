"""Create a full-scale composite from a pilot and its large-n extension.

The duplicate n=100000 frozen rows are verified before merging.  The plot uses
world-cluster bootstrap intervals, retaining all streams from a resampled world.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_sample_size_stress_test import generate_world


DEFAULT_ROOT = Path("/Users/cispa/Documents/OffCEM/sample_size_stress_results")
KEYS = [
    "world_seed",
    "stream_seed",
    "stream_index",
    "model_seed",
    "estimator",
    "partition",
]
METRICS = {
    "rel_mse": "Relative MSE",
    "bias2": "Within-world Bias squared",
    "variance": "Within-world Stream Variance",
}
ORDER = ["OffCEM matched", "OffCEM wfss", "OffCEM kmeans", "DR", "DM"]
COLORS = {
    "OffCEM matched": "#1f77b4",
    "OffCEM wfss": "#ff7f0e",
    "OffCEM kmeans": "#2ca02c",
    "DR": "#d62728",
    "DM": "#9467bd",
}
COVERAGE_METRICS = {
    "target_mass_on_observed_actions": (
        "Target-policy mass on observed actions",
        "#1f77b4",
    ),
    "target_best_action_observation_rate": (
        "Target-best-action observation rate",
        "#ff7f0e",
    ),
    "user_action_coverage": (
        "Observed user-action cells",
        "#2ca02c",
    ),
}


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
        default=DEFAULT_ROOT / "sample_size_frozen_full_scale",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def _world_metric_values(rows: pd.DataFrame, metric: str) -> np.ndarray:
    by_world = []
    for _, world_rows in rows.groupby("world_seed"):
        error = world_rows["error"].to_numpy(dtype=float)
        if metric == "rel_mse":
            value = np.mean(error**2 / world_rows["true_value"].to_numpy(dtype=float) ** 2)
        elif metric == "bias2":
            value = np.mean(error) ** 2
        elif metric == "variance":
            value = np.mean((error - error.mean()) ** 2)
        else:
            raise ValueError(metric)
        by_world.append(value)
    return np.asarray(by_world, dtype=float)


def _metric_from_rows(rows: pd.DataFrame, metric: str) -> float:
    return float(_world_metric_values(rows, metric).mean())


def standard_error(rows: pd.DataFrame, metric: str):
    """Return the mean and one standard error across independent worlds."""
    world_values = _world_metric_values(rows, metric)
    point = float(world_values.mean())
    if len(world_values) <= 1:
        return point, 0.0
    return point, float(world_values.std(ddof=1) / np.sqrt(len(world_values)))


def bootstrap_interval(rows: pd.DataFrame, metric: str, n_bootstrap: int, seed: int):
    world_values = _world_metric_values(rows, metric)
    point = _metric_from_rows(rows, metric)
    if len(world_values) <= 1:
        return point, point, point
    rng = np.random.RandomState(seed)
    sampled_idx = rng.randint(
        0, len(world_values), size=(n_bootstrap, len(world_values))
    )
    draws = world_values[sampled_idx].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_world_mean(
    rows: pd.DataFrame,
    column: str,
    n_bootstrap: int,
    seed: int,
):
    """Bootstrap an evaluation-stream diagnostic by resampling worlds."""
    world_values = rows.groupby("world_seed")[column].mean().to_numpy(dtype=float)
    point = float(world_values.mean())
    if len(world_values) <= 1:
        return point, point, point
    rng = np.random.RandomState(seed)
    sampled_idx = rng.randint(
        0, len(world_values), size=(n_bootstrap, len(world_values))
    )
    draws = world_values[sampled_idx].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)


def load_and_merge(pilot_dir: Path, extended_dir: Path) -> pd.DataFrame:
    pilot = pd.read_csv(pilot_dir / "sample_size_stress_tidy.csv")
    extended = pd.read_csv(extended_dir / "sample_size_stress_tidy.csv")
    pilot = pilot[pilot["arm"] == "frozen"].copy()
    extended = extended[extended["arm"] == "frozen"].copy()
    checkpoint = 100000
    left = pilot[pilot["n"] == checkpoint]
    right = extended[extended["n"] == checkpoint]
    joined = left.merge(right, on=KEYS, suffixes=("_pilot", "_extended"), how="outer", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("pilot and extension do not contain the same 100k frozen rows")
    for column in ("estimate", "true_value", "error", "squared_error"):
        if not np.allclose(
            joined[f"{column}_pilot"], joined[f"{column}_extended"], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"100k overlap check failed for {column}")
    return pd.concat(
        [pilot[pilot["n"] < checkpoint], extended], ignore_index=True
    )


def make_metric_plot(
    rows: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    n_bootstrap: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    estimators = [item for item in ORDER if item in set(rows["estimator"])]
    for estimator_index, estimator in enumerate(estimators):
        estimator_rows = rows[rows["estimator"] == estimator]
        x_values, points, standard_errors = [], [], []
        for n, n_rows in estimator_rows.groupby("n"):
            point, standard_error_value = standard_error(n_rows, metric)
            x_values.append(n)
            points.append(point)
            standard_errors.append(standard_error_value)
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
    axis.set_yscale("log")
    axis.set_xlabel("logged interactions n")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    output = out_dir / filename
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def make_empirical_coverage_plot(
    rows: pd.DataFrame,
    out_dir: Path,
    n_bootstrap: int,
) -> Path:
    """Show finite-sample coverage, not the unchanged population positivity."""
    diagnostics = rows[rows["estimator"] == "OffCEM matched"].copy()
    if diagnostics.empty:
        raise ValueError("matched OffCEM rows are required for coverage diagnostics")
    if diagnostics[list(COVERAGE_METRICS)].isna().any().any():
        raise ValueError("coverage diagnostics are missing from matched OffCEM rows")

    fig, axis = plt.subplots(figsize=(8, 5))
    for metric_index, (column, (label, color)) in enumerate(COVERAGE_METRICS.items()):
        x_values, points, lower, upper = [], [], [], []
        for n, n_rows in diagnostics.groupby("n"):
            point, low, high = bootstrap_world_mean(
                n_rows,
                column,
                n_bootstrap,
                seed=1000 * metric_index + int(n),
            )
            x_values.append(n)
            points.append(point)
            lower.append(low)
            upper.append(high)
        order = np.argsort(x_values)
        x_values = np.asarray(x_values)[order]
        points = np.asarray(points)[order]
        lower = np.asarray(lower)[order]
        upper = np.asarray(upper)[order]
        axis.plot(
            x_values,
            points,
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=label,
        )
        axis.fill_between(x_values, lower, upper, color=color, alpha=0.14, linewidth=0)

    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("logged evaluation interactions n")
    axis.set_ylabel("fraction covered")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    output = out_dir / "empirical_coverage_vs_n.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _population_overlap_metrics(world_seed: int) -> dict:
    """Compute exact support and importance-weight diagnostics for one world."""
    world = generate_world(
        world_seed=world_seed,
        n_clusters=50,
        eps=0.2,
        reward_std=3.0,
    )
    pi0 = world["pi_b_population"][:, :, 0]
    pi_e = world["pi_e_population"][:, :, 0]
    action_ratio = pi_e / pi0
    labels = world["cluster_indices"]
    pi0_cluster = np.column_stack(
        [pi0[:, labels == cluster].sum(axis=1) for cluster in range(50)]
    )
    pi_e_cluster = np.column_stack(
        [pi_e[:, labels == cluster].sum(axis=1) for cluster in range(50)]
    )
    cluster_ratio = pi_e_cluster / pi0_cluster
    return {
        "action_support_fraction": float(np.mean(pi0[pi_e > 0.0] > 0.0)),
        "cluster_support_fraction": float(
            np.mean(pi0_cluster[pi_e_cluster > 0.0] > 0.0)
        ),
        "action_max_ratio": float(action_ratio.max()),
        "cluster_max_ratio": float(cluster_ratio.max()),
        "action_ess_fraction": float(
            1.0 / np.mean(np.sum(pi0 * action_ratio**2, axis=1))
        ),
        "cluster_ess_fraction": float(
            1.0 / np.mean(np.sum(pi0_cluster * cluster_ratio**2, axis=1))
        ),
    }


def make_population_positivity_overlap_plot(
    rows: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Contrast fixed formal positivity with fixed practical overlap quality."""
    metrics = pd.DataFrame(
        [_population_overlap_metrics(int(seed)) for seed in sorted(rows["world_seed"].unique())]
    )
    labels = ["Action level", "Matched cluster level"]
    panels = [
        ("Formal support", "fraction with positive logging mass", "linear", [
            "action_support_fraction", "cluster_support_fraction",
        ]),
        ("Largest importance ratio", r"max $\pi_e / \pi_0$", "log", [
            "action_max_ratio", "cluster_max_ratio",
        ]),
        ("Importance-weight ESS fraction", "ESS / n", "log", [
            "action_ess_fraction", "cluster_ess_fraction",
        ]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, ylabel, scale, columns) in zip(axes, panels):
        values = [metrics[column].mean() for column in columns]
        axis.bar(range(2), values, color=["#d62728", "#1f77b4"], alpha=0.78)
        for index, column in enumerate(columns):
            axis.scatter(
                np.full(len(metrics), index),
                metrics[column],
                color="black",
                s=18,
                zorder=3,
            )
        axis.set_xticks(range(2), labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_yscale(scale)
        if scale == "linear":
            axis.set_ylim(0.0, 1.03)
        axis.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    output = out_dir / "population_positivity_and_overlap.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    rows = load_and_merge(args.pilot_dir, args.extended_dir)
    rel_mse = make_metric_plot(
        rows,
        args.out_dir,
        metric="rel_mse",
        ylabel="Relative MSE",
        filename="rel_mse_vs_n.png",
        n_bootstrap=args.n_bootstrap,
    )
    bias2 = make_metric_plot(
        rows,
        args.out_dir,
        metric="bias2",
        ylabel="Bias^2",
        filename="bias2_vs_n.png",
        n_bootstrap=args.n_bootstrap,
    )
    coverage = make_empirical_coverage_plot(rows, args.out_dir, args.n_bootstrap)
    positivity_overlap = make_population_positivity_overlap_plot(rows, args.out_dir)
    print(f"wrote {rel_mse}")
    print(f"wrote {bias2}")
    print(f"wrote {coverage}")
    print(f"wrote {positivity_overlap}")


if __name__ == "__main__":
    main()
