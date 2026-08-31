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
        x_values, points, lower, upper = [], [], [], []
        for n, n_rows in estimator_rows.groupby("n"):
            point, low, high = bootstrap_interval(
                n_rows,
                metric,
                n_bootstrap,
                seed=1000 * estimator_index + int(n),
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
            color=COLORS[estimator],
            label=estimator,
        )
        axis.fill_between(
            x_values,
            np.maximum(lower, 1e-14),
            np.maximum(upper, 1e-14),
            color=COLORS[estimator],
            alpha=0.14,
            linewidth=0,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("logged interactions n")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    output = out_dir / filename
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
    print(f"wrote {rel_mse}")
    print(f"wrote {bias2}")


if __name__ == "__main__":
    main()
