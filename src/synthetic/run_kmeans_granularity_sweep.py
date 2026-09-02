"""K-means-only OffCEM granularity sweep over logged sample size and K.

The synthetic reward world always uses 50 generating clusters.  For each
logged prefix, this runner evaluates only OffCEM with K-means action clusters;
no matched OffCEM, WFSS, DR, or DM models are fit.
"""
import argparse
import csv
import json
from pathlib import Path
from time import time
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from ope import train_reward_model_via_two_stage
from run_policy_disagreement_sweep import within_cluster_policy_metrics
from run_sample_size_stress_test import DEFAULTS
from run_sample_size_stress_test import _coverage_diagnostics
from run_sample_size_stress_test import _local_correctness_diagnostics
from run_sample_size_stress_test import _set_model_seed
from run_sample_size_stress_test import build_feedback_prefix
from run_sample_size_stress_test import estimate_offcem_compact
from run_sample_size_stress_test import generate_world
from run_sample_size_stress_test import parse_n_list
from run_sample_size_stress_test import sample_logged_stream_with_progress
from run_sample_size_stress_test import write_csv


DEFAULT_K_GRID = "500:10,30,50;1000:10,30,50;3000:30,50,70;10000:50,100,150;30000:50,100,150,200;100000:50,250,350,450"
ESTIMATOR_NAME = "OffCEM K-means"


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-means-only OffCEM granularity sweep"
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="500,1000,3000,10000,30000,100000",
    )
    parser.add_argument(
        "--kmeans-k-grid",
        type=str,
        default=DEFAULT_K_GRID,
        help="Semicolon-separated n:K,K entries; e.g. 500:10,30,50;1000:10,30,50",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-streams", type=int, default=1)
    parser.add_argument("--generating-n-clusters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument("--base-model-seed", type=int, default=400000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("kmeans_granularity_results"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_kmeans_k_grid(value: str, n_list: List[int], n_actions: int) -> Dict[int, List[int]]:
    """Parse and validate a per-sample-size K-means grid."""
    grid = {}
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        try:
            raw_n, raw_k_values = entry.split(":", maxsplit=1)
            n = int(raw_n.replace("_", ""))
            k_values = sorted(
                {
                    int(raw_k.replace("_", ""))
                    for raw_k in raw_k_values.split(",")
                    if raw_k.strip()
                }
            )
        except ValueError as error:
            raise ValueError(
                "--kmeans-k-grid entries must have the form n:K,K;..."
            ) from error
        if not k_values:
            raise ValueError(f"no K values supplied for n={n}")
        if any(k < 2 or k > n_actions for k in k_values):
            raise ValueError(
                f"K values for n={n} must lie in [2, {n_actions}]"
            )
        if n in grid:
            raise ValueError(f"duplicate K-means grid entry for n={n}")
        grid[n] = k_values
    if set(grid) != set(n_list):
        missing = sorted(set(n_list) - set(grid))
        extra = sorted(set(grid) - set(n_list))
        raise ValueError(
            f"K-means grid must cover exactly --n-list; missing={missing}, extra={extra}"
        )
    return grid


def result_row(
    prefix: Dict,
    world: Dict,
    stream_index: int,
    model_seed: int,
    kmeans_k: int,
    estimate: float,
    diagnostics: Dict[str, float],
    policy_metrics: Dict[str, float],
) -> Dict:
    true_value = float(world["true_value"])
    error = float(estimate - true_value)
    return {
        "n": int(prefix["n_rounds"]),
        "world_seed": int(world["world_seed"]),
        "stream_seed": int(prefix["stream_seed"]),
        "stream_index": int(stream_index),
        "model_seed": int(model_seed),
        "estimator": ESTIMATOR_NAME,
        "partition": "kmeans",
        "kmeans_k": int(kmeans_k),
        "estimate": float(estimate),
        "true_value": true_value,
        "error": error,
        "squared_error": float(error**2),
        "within_cluster_tv": float(policy_metrics["within_cluster_tv"]),
        "within_cluster_chi2": float(policy_metrics["within_cluster_chi2"]),
        "pairwise_ratio_mse": float(policy_metrics["pairwise_ratio_mse"]),
        **diagnostics,
    }


def run_world(
    world_index: int,
    n_list: List[int],
    k_grid: Dict[int, List[int]],
    generating_n_clusters: int,
    eps: float,
    reward_std: float,
    n_streams: int,
    base_model_seed: int,
    progress,
    checkpoint_callback: Optional[Callable[[List[Dict]], None]] = None,
) -> List[Dict]:
    world_seed = DEFAULTS["RANDOM_STATE"] + world_index
    world = generate_world(
        world_seed=world_seed,
        n_clusters=generating_n_clusters,
        eps=eps,
        reward_std=reward_std,
    )
    labels_by_k = {}
    policy_metrics_by_k = {}
    partition_seed = world_seed + DEFAULTS["ESTIMATION_SEED_OFFSET"]
    for kmeans_k in sorted({k for values in k_grid.values() for k in values}):
        labels = compute_clusters(
            world["action_context_one_hot"],
            n_clusters=kmeans_k,
            method="kmeans",
            balance="natural",
            random_state=partition_seed + kmeans_k,
        )
        labels_by_k[kmeans_k] = labels
        policy_metrics_by_k[kmeans_k] = within_cluster_policy_metrics(
            pi0_population=world["pi_b_population"],
            pi_lambda=world["pi_e_population"],
            cluster_labels=labels,
        )

    rows = []
    for stream_index in range(n_streams):
        progress.set_postfix_str(
            f"world={world_index + 1} stream={stream_index + 1}/{n_streams} sampling {max(n_list)} rounds"
        )
        full_stream = sample_logged_stream_with_progress(
            world=world,
            n_rounds=max(n_list),
            stream_seed=world_seed + DEFAULTS["STREAM_SEED_OFFSET"] + stream_index,
            reward_std=reward_std,
            compact=True,
            progress_desc=f"w{world_index + 1} stream {stream_index + 1}/{n_streams} logging",
            show_progress=True,
        )
        for n_index, n in enumerate(n_list):
            prefix = build_feedback_prefix(full_stream, n)
            for kmeans_k in k_grid[n]:
                model_seed = int(
                    base_model_seed
                    + world_index * 1_000_000
                    + stream_index * 100_000
                    + n_index * 1_000
                    + kmeans_k
                )
                labels = labels_by_k[kmeans_k]
                progress.set_postfix_str(
                    f"world={world_index + 1} stream={stream_index + 1}/{n_streams} n={n} K={kmeans_k}"
                )
                _set_model_seed(model_seed)
                f_population = train_reward_model_via_two_stage(
                    bandit_data=prefix,
                    clusters=clusters_to_onehot_3d(labels, int(world["n_users"])),
                    need_q_x_a=False,
                    random_state=model_seed,
                    prediction_context=world["fixed_user_contexts"],
                    progress_desc=(
                        f"w{world_index + 1} s{stream_index + 1} n{n} K{kmeans_k} pairwise"
                    ),
                )
                diagnostics = _local_correctness_diagnostics(world, f_population, labels)
                diagnostics.update(_coverage_diagnostics(prefix, world, labels))
                rows.append(
                    result_row(
                        prefix=prefix,
                        world=world,
                        stream_index=stream_index,
                        model_seed=model_seed,
                        kmeans_k=kmeans_k,
                        estimate=estimate_offcem_compact(
                            prefix, world, labels, f_population
                        ),
                        diagnostics=diagnostics,
                        policy_metrics=policy_metrics_by_k[kmeans_k],
                    )
                )
                progress.update(1)
                if checkpoint_callback is not None:
                    checkpoint_callback(rows)
    return rows


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["n"]), int(row["kmeans_k"])), []).append(row)
    aggregates = []
    for (n, kmeans_k), items in sorted(grouped.items()):
        errors = np.asarray([item["error"] for item in items], dtype=float)
        true_values = np.asarray([item["true_value"] for item in items], dtype=float)
        aggregate = {
            "n": n,
            "kmeans_k": kmeans_k,
            "estimator": ESTIMATOR_NAME,
            "mse": float(np.mean(errors**2)),
            "rel_mse": float(np.mean(errors**2 / np.maximum(true_values**2, 1e-12))),
            "bias2": float(np.mean(errors) ** 2),
            "variance": float(np.mean((errors - errors.mean()) ** 2)),
            "estimate_mean": float(np.mean([item["estimate"] for item in items])),
            "true_value_mean": float(true_values.mean()),
            "n_worlds": len({item["world_seed"] for item in items}),
            "n_rows": len(items),
        }
        for key in (
            "within_cluster_tv",
            "within_cluster_chi2",
            "pairwise_ratio_mse",
            "reward_mse",
            "lc_error",
            "ari_to_matched",
            "cluster_size_min",
            "cluster_size_max",
            "cluster_size_std",
            "cluster_ess_fraction",
            "cluster_weight_max",
            "cluster_weight_variance",
            "pairwise_training_examples",
            "target_mass_on_observed_actions",
            "user_action_coverage",
        ):
            aggregate[f"{key}_mean"] = float(np.mean([item[key] for item in items]))
        aggregates.append(aggregate)
    return aggregates


def load_tidy_csv(path: Path) -> List[Dict]:
    with open(path) as file:
        rows = []
        for raw in csv.DictReader(file):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = int(value) if key in {"n", "kmeans_k", "world_seed", "stream_seed", "stream_index", "model_seed"} else float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def plot_aggregates(aggregates: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for kmeans_k in sorted({row["kmeans_k"] for row in aggregates}):
        selected = sorted(
            [row for row in aggregates if row["kmeans_k"] == kmeans_k],
            key=lambda row: row["n"],
        )
        axis.plot(
            [row["n"] for row in selected],
            [row["rel_mse"] for row in selected],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=f"K={kmeans_k}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("logged interactions n")
    axis.set_ylabel("Relative MSE")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "rel_mse_vs_n_by_k.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args) -> None:
    if args.quick:
        args.n_list = "500,3000"
        args.kmeans_k_grid = "500:10,30,50;3000:30,50,70"
        args.n_seeds = 2
        args.n_streams = 1
        print("[quick] n-list=500,3000 n-seeds=2")
    n_list = parse_n_list(args.n_list)
    k_grid = parse_kmeans_k_grid(
        args.kmeans_k_grid,
        n_list=n_list,
        n_actions=DEFAULTS["N_ACTIONS"],
    )
    if args.n_seeds <= 0 or args.n_streams <= 0:
        raise ValueError("--n-seeds and --n-streams must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "kmeans_granularity_tidy.csv"
    aggregate_path = out_dir / "kmeans_granularity_aggregate.csv"
    if args.plot_only:
        rows = load_tidy_csv(tidy_path)
        aggregates = aggregate_results(rows)
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir)
        return

    total_tasks = args.n_seeds * args.n_streams * sum(
        len(k_grid[n]) for n in n_list
    )
    all_rows = []
    started = time()
    progress = tqdm(total=total_tasks, desc="K-means granularity models")
    for world_index in range(args.n_seeds):
        progress.set_postfix_str(f"world={world_index + 1} initializing")

        def checkpoint(world_rows):
            with open(out_dir / "latest_rows.json", "w") as file:
                json.dump(all_rows + world_rows, file, indent=2)

        all_rows.extend(
            run_world(
                world_index=world_index,
                n_list=n_list,
                k_grid=k_grid,
                generating_n_clusters=args.generating_n_clusters,
                eps=args.eps,
                reward_std=args.reward_std,
                n_streams=args.n_streams,
                base_model_seed=args.base_model_seed,
                progress=progress,
                checkpoint_callback=checkpoint,
            )
        )
    progress.close()
    aggregates = aggregate_results(all_rows)
    write_csv(tidy_path, all_rows)
    write_csv(aggregate_path, aggregates)
    if not args.no_plot:
        plot_aggregates(aggregates, out_dir)
    print(f"wrote {tidy_path}")
    print(f"wrote {aggregate_path}")
    print(f"done in {(time() - started) / 60:.1f} min")


if __name__ == "__main__":
    run_experiment(parse_args())
