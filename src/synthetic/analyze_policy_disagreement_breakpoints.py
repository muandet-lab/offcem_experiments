"""Analyze policy-disagreement sweep outputs into failure boundaries.

This is a post-processing script: it reads an existing
``policy_disagreement_sweep_tidy.csv`` and does not rerun OPE models.

Example:

    conda run -n offcem python -m analyze_policy_disagreement_breakpoints \
      --results-dir /Users/cispa/Documents/OffCEM/policy_disagreement_results/policy_disagreement_failure_map
"""
import argparse
import csv
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTITION_ORDER = [
    "matched",
    "wfss",
    "kmeans",
    "spectral",
    "agglomerative",
    "dps",
    "random",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build tolerance-aware OffCEM failure boundaries"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/policy_disagreement_results/policy_disagreement_failure_map",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.10,
        help="Material-loss threshold: OffCEM MSE > (1 + delta) DR MSE",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_tidy(path: Path) -> List[Dict]:
    with open(path) as file:
        return [
            {
                "seed": int(row["seed"]),
                "partition": row["partition"],
                "lambda": float(row["lambda"]),
                "tau": float(row["tau"]),
                "sq_error_offcem": float(row["sq_error_offcem"]),
                "sq_error_dr": float(row["sq_error_dr"]),
                "within_cluster_tv": float(row["within_cluster_tv"]),
                "within_cluster_chi2": float(row["within_cluster_chi2"]),
                "pairwise_ratio_mse": float(row["pairwise_ratio_mse"]),
                "lc_dm_mse_pi0": float(row["lc_dm_mse_pi0"]),
                "pairwise_lc_mse": float(row["pairwise_lc_mse"]),
                "theorem33_bias": float(row["theorem33_bias"]),
                "ARI_to_generating_partition": float(
                    row["ARI_to_generating_partition"]
                ),
            }
            for row in csv.DictReader(file)
        ]


def paired_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.RandomState,
) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or n_bootstrap <= 0:
        return float(values.mean()), float(values.mean())
    indices = rng.randint(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def summarize_cells(
    rows: Iterable[Dict],
    delta: float,
    bootstrap_samples: int,
    alpha: float,
    random_state: int,
) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["partition"], row["lambda"], row["tau"])
        grouped.setdefault(key, []).append(row)

    rng = np.random.RandomState(random_state)
    cells = []
    for (partition, lambda_, tau), items in sorted(grouped.items()):
        diff = np.array(
            [
                item["sq_error_offcem"] - (1.0 + delta) * item["sq_error_dr"]
                for item in items
            ],
            dtype=float,
        )
        ci_low, ci_high = paired_bootstrap_ci(
            diff,
            n_bootstrap=bootstrap_samples,
            alpha=alpha,
            rng=rng,
        )
        cells.append(
            {
                "partition": partition,
                "lambda": float(lambda_),
                "tau": float(tau),
                "n_seeds": len(items),
                "mse_offcem": float(
                    np.mean([item["sq_error_offcem"] for item in items])
                ),
                "mse_dr": float(np.mean([item["sq_error_dr"] for item in items])),
                "material_mse_diff": float(diff.mean()),
                "material_mse_diff_ci_low": ci_low,
                "material_mse_diff_ci_high": ci_high,
                "material_loss_mean": bool(diff.mean() > 0.0),
                "material_loss_ci": bool(ci_low > 0.0),
                "within_cluster_tv": float(
                    np.mean([item["within_cluster_tv"] for item in items])
                ),
                "within_cluster_chi2": float(
                    np.mean([item["within_cluster_chi2"] for item in items])
                ),
                "pairwise_ratio_mse": float(
                    np.mean([item["pairwise_ratio_mse"] for item in items])
                ),
                "lc_dm_mse_pi0": float(
                    np.mean([item["lc_dm_mse_pi0"] for item in items])
                ),
                "pairwise_lc_mse": float(
                    np.mean([item["pairwise_lc_mse"] for item in items])
                ),
                "theorem33_abs_bias": float(
                    np.mean([abs(item["theorem33_bias"]) for item in items])
                ),
                "ARI_to_generating_partition": float(
                    np.mean(
                        [item["ARI_to_generating_partition"] for item in items]
                    )
                ),
            }
        )
    return cells


def interpolate_metric(
    previous: Optional[Dict],
    current: Dict,
    metric: str,
    y_key: str = "material_mse_diff",
) -> float:
    if previous is None:
        return float(current[metric])
    y0 = float(previous[y_key])
    y1 = float(current[y_key])
    x0 = float(previous[metric])
    x1 = float(current[metric])
    if y1 == y0:
        return x1
    t = (0.0 - y0) / (y1 - y0)
    t = min(max(t, 0.0), 1.0)
    return float(x0 + t * (x1 - x0))


def make_boundary_rows(cells: List[Dict]) -> List[Dict]:
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell["partition"], cell["tau"]), []).append(cell)

    rows = []
    for (partition, tau), items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: row["lambda"])
        lc_error = float(np.mean([row["lc_dm_mse_pi0"] for row in items]))
        pairwise_lc = float(np.mean([row["pairwise_lc_mse"] for row in items]))
        ari = float(np.mean([row["ARI_to_generating_partition"] for row in items]))
        first_mean = _first_crossing(items, "material_loss_mean")
        first_ci = _first_crossing(items, "material_loss_ci")
        rows.append(
            {
                "partition": partition,
                "tau": float(tau),
                "L_lc_dm_mse_pi0": lc_error,
                "L_pairwise_lc_mse": pairwise_lc,
                "ARI_to_generating_partition": ari,
                **_breakpoint_fields(items, first_mean, "mean"),
                **_breakpoint_fields(items, first_ci, "ci"),
            }
        )
    return rows


def _first_crossing(items: List[Dict], key: str) -> Optional[Dict]:
    for row in items:
        if row[key]:
            return row
    return None


def _previous_row(items: List[Dict], current: Dict) -> Optional[Dict]:
    index = items.index(current)
    return items[index - 1] if index > 0 else None


def _breakpoint_fields(
    items: List[Dict],
    crossing: Optional[Dict],
    suffix: str,
) -> Dict:
    prefix = f"breakpoint_{suffix}"
    if crossing is None:
        return {
            f"{prefix}_exists": False,
            f"{prefix}_lambda": np.nan,
            f"{prefix}_D_chi2": np.nan,
            f"{prefix}_D_tv": np.nan,
            f"{prefix}_pairwise_ratio_mse": np.nan,
            f"{prefix}_material_mse_diff": np.nan,
            f"{prefix}_material_mse_diff_ci_low": np.nan,
            f"{prefix}_material_mse_diff_ci_high": np.nan,
        }
    previous = _previous_row(items, crossing)
    y_key = (
        "material_mse_diff_ci_low"
        if suffix == "ci"
        else "material_mse_diff"
    )
    return {
        f"{prefix}_exists": True,
        f"{prefix}_lambda": interpolate_metric(
            previous,
            crossing,
            "lambda",
            y_key=y_key,
        ),
        f"{prefix}_D_chi2": interpolate_metric(
            previous,
            crossing,
            "within_cluster_chi2",
            y_key=y_key,
        ),
        f"{prefix}_D_tv": interpolate_metric(
            previous,
            crossing,
            "within_cluster_tv",
            y_key=y_key,
        ),
        f"{prefix}_pairwise_ratio_mse": interpolate_metric(
            previous,
            crossing,
            "pairwise_ratio_mse",
            y_key=y_key,
        ),
        f"{prefix}_material_mse_diff": float(crossing["material_mse_diff"]),
        f"{prefix}_material_mse_diff_ci_low": float(
            crossing["material_mse_diff_ci_low"]
        ),
        f"{prefix}_material_mse_diff_ci_high": float(
            crossing["material_mse_diff_ci_high"]
        ),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_boundary(boundary_rows: List[Dict], out_dir: Path, delta: float) -> None:
    ordered = sorted(
        boundary_rows,
        key=lambda row: (
            PARTITION_ORDER.index(row["partition"])
            if row["partition"] in PARTITION_ORDER
            else 999,
            row["partition"],
        ),
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row in ordered:
        x = row["L_lc_dm_mse_pi0"]
        y = row["breakpoint_ci_D_chi2"]
        has_ci = bool(row["breakpoint_ci_exists"])
        if not has_ci:
            y = row["breakpoint_mean_D_chi2"]
        if np.isnan(y):
            y = 0.0
        ax.scatter(
            [x],
            [y],
            s=80,
            marker="o" if has_ci else "x",
            color="#1f77b4" if has_ci else "#d62728",
        )
        ax.annotate(
            row["partition"],
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("L: pi0-centered local-correctness MSE")
    ax.set_ylabel("D_chi2*: material OffCEM-vs-DR breakpoint")
    ax.set_title(f"Failure boundary, delta={delta:g}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"tolerance_failure_boundary_delta{delta:g}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.results_dir)
    tidy_path = out_dir / "policy_disagreement_sweep_tidy.csv"
    rows = load_tidy(tidy_path)
    cells = summarize_cells(
        rows,
        delta=args.delta,
        bootstrap_samples=args.bootstrap_samples,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    boundary = make_boundary_rows(cells)
    delta_tag = f"{args.delta:g}"
    cell_path = out_dir / f"tolerance_breakpoint_cells_delta{delta_tag}.csv"
    boundary_path = out_dir / f"tolerance_breakpoint_summary_delta{delta_tag}.csv"
    write_csv(cell_path, cells)
    write_csv(boundary_path, boundary)
    if not args.no_plot:
        plot_boundary(boundary, out_dir, delta=args.delta)
    print(f"wrote {cell_path}")
    print(f"wrote {boundary_path}")
    if not args.no_plot:
        print(f"wrote {out_dir / f'tolerance_failure_boundary_delta{delta_tag}.png'}")


if __name__ == "__main__":
    main()
