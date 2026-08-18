"""Analyze whether signed policy-LC alignment explains OffCEM failure.

This post-processes an existing policy-disagreement sweep. It does not rerun
any estimators.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


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
        description="Analyze policy-LC signed alignment in a completed sweep"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/policy_disagreement_results/policy_disagreement_failure_map",
    )
    parser.add_argument("--delta", type=float, default=0.10)
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
                "error_offcem": float(row["error_offcem"]),
                "sq_error_offcem": float(row["sq_error_offcem"]),
                "sq_error_dr": float(row["sq_error_dr"]),
                "within_cluster_tv": float(row["within_cluster_tv"]),
                "within_cluster_chi2": float(row["within_cluster_chi2"]),
                "lc_dm_mse_pi0": float(row["lc_dm_mse_pi0"]),
                "pairwise_lc_mse": float(row["pairwise_lc_mse"]),
                "policy_lc_weighted_covariance": float(
                    row["policy_lc_weighted_covariance"]
                ),
                "theorem33_bias": float(row["theorem33_bias"]),
            }
            for row in csv.DictReader(file)
        ]


def summarize_alignment(rows: Iterable[Dict], delta: float) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["partition"], row["lambda"], row["tau"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (partition, lambda_, tau), items in sorted(grouped.items()):
        offcem_mse = float(np.mean([item["sq_error_offcem"] for item in items]))
        dr_mse = float(np.mean([item["sq_error_dr"] for item in items]))
        covariance = float(
            np.mean([item["policy_lc_weighted_covariance"] for item in items])
        )
        theorem_bias = float(np.mean([item["theorem33_bias"] for item in items]))
        empirical_bias = float(np.mean([item["error_offcem"] for item in items]))
        summaries.append(
            {
                "partition": partition,
                "lambda": float(lambda_),
                "tau": float(tau),
                "n_seeds": len(items),
                "lc_dm_mse_pi0": float(
                    np.mean([item["lc_dm_mse_pi0"] for item in items])
                ),
                "pairwise_lc_mse": float(
                    np.mean([item["pairwise_lc_mse"] for item in items])
                ),
                "within_cluster_tv": float(
                    np.mean([item["within_cluster_tv"] for item in items])
                ),
                "within_cluster_chi2": float(
                    np.mean([item["within_cluster_chi2"] for item in items])
                ),
                "policy_lc_weighted_covariance": covariance,
                "abs_policy_lc_weighted_covariance": float(abs(covariance)),
                "theorem33_bias": theorem_bias,
                "abs_theorem33_bias": float(abs(theorem_bias)),
                "empirical_offcem_bias": empirical_bias,
                "abs_empirical_offcem_bias": float(abs(empirical_bias)),
                "mse_offcem": offcem_mse,
                "mse_dr": dr_mse,
                "material_mse_diff_vs_dr": float(
                    offcem_mse - (1.0 + delta) * dr_mse
                ),
                "materially_worse_than_dr": bool(
                    offcem_mse > (1.0 + delta) * dr_mse
                ),
            }
        )
    return summaries


def final_lambda_rows(summaries: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in summaries:
        grouped.setdefault((row["partition"], row["tau"]), []).append(row)
    final = []
    for _, rows in sorted(grouped.items()):
        final.append(max(rows, key=lambda row: row["lambda"]))
    return sorted(final, key=partition_sort_key)


def correlation_rows(summaries: List[Dict]) -> List[Dict]:
    rows = []
    nonzero = [row for row in summaries if row["lambda"] > 0.0]
    for y_key in (
        "mse_offcem",
        "abs_empirical_offcem_bias",
        "material_mse_diff_vs_dr",
    ):
        for x_key in (
            "lc_dm_mse_pi0",
            "within_cluster_chi2",
            "abs_policy_lc_weighted_covariance",
            "abs_theorem33_bias",
        ):
            xs = np.array([row[x_key] for row in nonzero], dtype=float)
            ys = np.array([row[y_key] for row in nonzero], dtype=float)
            rho, pvalue = spearmanr(xs, ys)
            rows.append(
                {
                    "subset": "lambda_gt_0",
                    "x": x_key,
                    "y": y_key,
                    "spearman": float(rho),
                    "pvalue": float(pvalue),
                    "n": len(nonzero),
                }
            )
    return rows


def partition_sort_key(row: Dict):
    partition = row["partition"]
    return (
        PARTITION_ORDER.index(partition)
        if partition in PARTITION_ORDER
        else len(PARTITION_ORDER),
        partition,
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_alignment(summaries: List[Dict], out_dir: Path) -> None:
    final = final_lambda_rows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        (
            "lc_dm_mse_pi0",
            "abs_empirical_offcem_bias",
            "LC error alone",
            "L: pi0-centered LC MSE",
        ),
        (
            "abs_policy_lc_weighted_covariance",
            "abs_empirical_offcem_bias",
            "Signed policy-LC alignment",
            "|policy-LC covariance|",
        ),
    ]
    for ax, (x_key, y_key, title, xlabel) in zip(axes, panels):
        for row in final:
            ax.scatter([row[x_key]], [row[y_key]], s=75, color="#1f77b4")
            ax.annotate(
                row["partition"],
                (row[x_key], row[y_key]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("abs empirical OffCEM bias at max lambda")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        out_dir / "policy_lc_alignment_final_bias.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for partition in PARTITION_ORDER:
        selected = [
            row for row in summaries if row["partition"] == partition
        ]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: row["within_cluster_chi2"])
        ax.plot(
            [row["within_cluster_chi2"] for row in selected],
            [row["abs_policy_lc_weighted_covariance"] for row in selected],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=partition,
        )
    ax.set_xlabel("D_chi2")
    ax.set_ylabel("|policy-LC covariance|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out_dir / "policy_lc_alignment_vs_chi2.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.results_dir)
    rows = load_tidy(out_dir / "policy_disagreement_sweep_tidy.csv")
    summaries = summarize_alignment(rows, delta=args.delta)
    final = final_lambda_rows(summaries)
    correlations = correlation_rows(summaries)
    write_csv(out_dir / "policy_lc_alignment_by_cell.csv", summaries)
    write_csv(out_dir / "policy_lc_alignment_final_lambda.csv", final)
    write_csv(out_dir / "policy_lc_alignment_correlations.csv", correlations)
    if not args.no_plot:
        plot_alignment(summaries, out_dir)
    print(f"wrote {out_dir / 'policy_lc_alignment_by_cell.csv'}")
    print(f"wrote {out_dir / 'policy_lc_alignment_final_lambda.csv'}")
    print(f"wrote {out_dir / 'policy_lc_alignment_correlations.csv'}")
    if not args.no_plot:
        print(f"wrote {out_dir / 'policy_lc_alignment_final_bias.png'}")
        print(f"wrote {out_dir / 'policy_lc_alignment_vs_chi2.png'}")


if __name__ == "__main__":
    main()
