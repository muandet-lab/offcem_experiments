"""End-to-end diagnostic for standard versus oracle-initialized K-means.

The synthetic reward world always has a 50-cluster WFSS generating partition.
For each requested logged prefix and K, this runner fits OffCEM twice using the
same fixed world, logged stream, and model RNG:

* standard K-means (K-means++ initialization);
* oracle-initialized K-means, whose starting centers are derived from the
  generating labels but whose final labels are still obtained by K-means on the
  observable action features.

This is deliberately an oracle diagnostic, not a deployable clustering method.
It separates poor K-means initialization from the structural mismatch between
the WFSS reward partition and the K-means feature geometry.
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
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from ope import train_reward_model_via_two_stage
from run_kmeans_granularity_sweep import parse_kmeans_k_grid
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


DEFAULT_K_BEST_GRID = "500:50;1000:30;3000:50;10000:150;30000:200;100000:450"
STANDARD = "standard"
ORACLE_INITIALIZED = "oracle_initialized"


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end standard versus oracle-initialized K-means diagnostic"
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="500,1000,3000,10000,30000,100000",
    )
    parser.add_argument(
        "--k-best-grid",
        type=str,
        default=DEFAULT_K_BEST_GRID,
        help="Semicolon-separated n:K entries selected from the prior granularity sweep",
    )
    parser.add_argument("--n-worlds", type=int, default=10)
    parser.add_argument("--n-streams", type=int, default=1)
    parser.add_argument(
        "--world-seed-base",
        type=int,
        default=22345,
        help="Fresh world-seed base; distinct from the earlier 12345..12354 sweep",
    )
    parser.add_argument("--generating-n-clusters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument(
        "--base-model-seed",
        type=int,
        default=DEFAULTS["MODEL_SEED_OFFSET"],
        help="Model RNG offset; fixed within a world across n, K, and initialization",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("kmeans_oracle_initialization_results"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_k_best_grid(value: str, n_list: List[int], n_actions: int) -> Dict[int, int]:
    """Parse one preselected K_best value for every requested sample size."""
    grid = parse_kmeans_k_grid(
        value,
        n_list=n_list,
        n_actions=n_actions,
    )
    if any(len(k_values) != 1 for k_values in grid.values()):
        raise ValueError("--k-best-grid must supply exactly one K value per n")
    return {n: k_values[0] for n, k_values in grid.items()}


def k_cases_for_n(k_best: int) -> List[int]:
    """Return the unique fixed-K and selected-K conditions for one n."""
    return sorted({50, int(k_best)})


def k_role(kmeans_k: int, k_best: int) -> str:
    if kmeans_k == 50 and kmeans_k == k_best:
        return "fixed_50_and_selected"
    if kmeans_k == 50:
        return "fixed_50"
    return "selected_k_best"


def generating_centroids(
    action_features: np.ndarray,
    generating_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return label ids, feature centroids, and action counts of generating groups."""
    label_ids = np.unique(generating_labels)
    centers = np.vstack(
        [action_features[generating_labels == label].mean(axis=0) for label in label_ids]
    )
    counts = np.asarray(
        [np.count_nonzero(generating_labels == label) for label in label_ids], dtype=int
    )
    return label_ids, centers, counts


def proportional_subcluster_counts(counts: np.ndarray, n_centers: int) -> np.ndarray:
    """Allocate at least one, and at most all actions, to each generating group."""
    counts = np.asarray(counts, dtype=int)
    if n_centers < len(counts) or n_centers > int(counts.sum()):
        raise ValueError("n_centers must lie between number of groups and actions")
    allocation = np.ones(len(counts), dtype=int)
    target = n_centers * counts / counts.sum()
    while allocation.sum() < n_centers:
        eligible = allocation < counts
        if not np.any(eligible):
            raise AssertionError("unable to allocate requested number of centers")
        deficits = target - allocation
        deficits[~eligible] = -np.inf
        allocation[int(np.argmax(deficits))] += 1
    return allocation


def oracle_initial_centers(
    action_features: np.ndarray,
    generating_labels: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    """Construct K centers from the known generating partition.

    K=number of generating groups uses their exact feature means.  For fewer
    centers, weighted K-means coarsens those means.  For more centers, each
    generating group is deterministically split internally before the final
    K-means fit.  All cases are oracle-informed initialization only.
    """
    action_features = np.asarray(action_features, dtype=float)
    generating_labels = np.asarray(generating_labels)
    label_ids, centers, counts = generating_centroids(action_features, generating_labels)
    n_generating = len(label_ids)

    if n_clusters == n_generating:
        return centers, "generating_centroids"

    if n_clusters < n_generating:
        coarsener = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        coarsener.fit(centers, sample_weight=counts)
        return coarsener.cluster_centers_, "oracle_coarsened_generating_centroids"

    allocation = proportional_subcluster_counts(counts, n_clusters)
    split_centers = []
    for label, n_subclusters in zip(label_ids, allocation):
        group_features = action_features[generating_labels == label]
        if n_subclusters == 1:
            split_centers.append(group_features.mean(axis=0, keepdims=True))
            continue
        splitter = KMeans(
            n_clusters=int(n_subclusters),
            random_state=int(random_state + label),
            n_init=10,
        )
        splitter.fit(group_features)
        split_centers.append(splitter.cluster_centers_)
    return np.vstack(split_centers), "oracle_split_generating_centroids"


def labels_from_centers(action_features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared_distances = ((action_features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return squared_distances.argmin(axis=1)


def centroid_displacement(initial_centers: np.ndarray, final_centers: np.ndarray) -> float:
    """RMS displacement after optimal center matching, invariant to label order."""
    squared_distances = ((initial_centers[:, None, :] - final_centers[None, :, :]) ** 2).sum(axis=2)
    initial_idx, final_idx = linear_sum_assignment(squared_distances)
    return float(np.sqrt(squared_distances[initial_idx, final_idx].mean()))


def fit_kmeans_partition(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int,
    initial_centers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one standard or explicitly initialized K-means partition."""
    if initial_centers is None:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    else:
        if initial_centers.shape != (n_clusters, action_features.shape[1]):
            raise ValueError("initial centers have an incompatible shape")
        model = KMeans(
            n_clusters=n_clusters,
            init=np.asarray(initial_centers, dtype=float),
            n_init=1,
            random_state=random_state,
        )
    labels = model.fit_predict(action_features)
    return labels.astype(int), model.cluster_centers_


def build_partitions(world: Dict, requested_k: Iterable[int], partition_seed: int) -> Dict:
    """Build standard and oracle-initialized K-means partitions once per world."""
    features = world["action_context_one_hot"]
    generating_labels = world["cluster_indices"]
    partitions = {}
    for kmeans_k in sorted(set(requested_k)):
        standard_labels, _ = fit_kmeans_partition(
            features,
            n_clusters=kmeans_k,
            random_state=partition_seed + kmeans_k,
        )
        partitions[(kmeans_k, STANDARD)] = {
            "labels": standard_labels,
            "initialization_type": "kmeans_plus_plus",
            "initial_ari_to_generating": np.nan,
            "final_centroid_displacement": np.nan,
        }

        initial_centers, initialization_type = oracle_initial_centers(
            features,
            generating_labels,
            n_clusters=kmeans_k,
            random_state=partition_seed + kmeans_k,
        )
        oracle_labels, final_centers = fit_kmeans_partition(
            features,
            n_clusters=kmeans_k,
            random_state=partition_seed + kmeans_k,
            initial_centers=initial_centers,
        )
        partitions[(kmeans_k, ORACLE_INITIALIZED)] = {
            "labels": oracle_labels,
            "initialization_type": initialization_type,
            "initial_ari_to_generating": float(
                adjusted_rand_score(
                    generating_labels,
                    labels_from_centers(features, initial_centers),
                )
            ),
            "final_centroid_displacement": centroid_displacement(
                initial_centers,
                final_centers,
            ),
        }
    return partitions


def result_row(
    prefix: Dict,
    world: Dict,
    stream_index: int,
    model_seed: int,
    kmeans_k: int,
    k_best: int,
    initialization: str,
    partition_info: Dict,
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
        "estimator": "OffCEM K-means",
        "initialization": initialization,
        "initialization_type": partition_info["initialization_type"],
        "kmeans_k": int(kmeans_k),
        "selected_k_best": int(k_best),
        "k_role": k_role(kmeans_k, k_best),
        "estimate": float(estimate),
        "true_value": true_value,
        "error": error,
        "squared_error": float(error**2),
        "initial_ari_to_generating": partition_info["initial_ari_to_generating"],
        "final_centroid_displacement": partition_info["final_centroid_displacement"],
        "within_cluster_tv": float(policy_metrics["within_cluster_tv"]),
        "within_cluster_chi2": float(policy_metrics["within_cluster_chi2"]),
        "pairwise_ratio_mse": float(policy_metrics["pairwise_ratio_mse"]),
        **diagnostics,
    }


def run_world(
    world_index: int,
    n_list: List[int],
    k_best_by_n: Dict[int, int],
    generating_n_clusters: int,
    eps: float,
    reward_std: float,
    n_streams: int,
    world_seed_base: int,
    base_model_seed: int,
    progress,
    checkpoint_callback: Optional[Callable[[List[Dict]], None]] = None,
) -> List[Dict]:
    world_seed = int(world_seed_base + world_index)
    world = generate_world(
        world_seed=world_seed,
        n_clusters=generating_n_clusters,
        eps=eps,
        reward_std=reward_std,
    )
    requested_k = [k for n in n_list for k in k_cases_for_n(k_best_by_n[n])]
    partitions = build_partitions(
        world,
        requested_k=requested_k,
        partition_seed=world_seed + DEFAULTS["ESTIMATION_SEED_OFFSET"],
    )
    policy_metrics = {
        key: within_cluster_policy_metrics(
            pi0_population=world["pi_b_population"],
            pi_lambda=world["pi_e_population"],
            cluster_labels=value["labels"],
        )
        for key, value in partitions.items()
    }
    model_seed = int(base_model_seed + world_index)
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
        for n in n_list:
            prefix = build_feedback_prefix(full_stream, n)
            for kmeans_k in k_cases_for_n(k_best_by_n[n]):
                for initialization in (STANDARD, ORACLE_INITIALIZED):
                    key = (kmeans_k, initialization)
                    partition_info = partitions[key]
                    labels = partition_info["labels"]
                    progress.set_postfix_str(
                        f"world={world_index + 1} stream={stream_index + 1}/{n_streams} "
                        f"n={n} K={kmeans_k} {initialization}"
                    )
                    _set_model_seed(model_seed)
                    f_population = train_reward_model_via_two_stage(
                        bandit_data=prefix,
                        clusters=clusters_to_onehot_3d(labels, int(world["n_users"])),
                        need_q_x_a=False,
                        random_state=model_seed,
                        prediction_context=world["fixed_user_contexts"],
                        progress_desc=(
                            f"w{world_index + 1} s{stream_index + 1} n{n} "
                            f"K{kmeans_k} {initialization} pairwise"
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
                            k_best=k_best_by_n[n],
                            initialization=initialization,
                            partition_info=partition_info,
                            estimate=estimate_offcem_compact(prefix, world, labels, f_population),
                            diagnostics=diagnostics,
                            policy_metrics=policy_metrics[key],
                        )
                    )
                    progress.update(1)
                    if checkpoint_callback is not None:
                        checkpoint_callback(rows)
    return rows


AGGREGATED_MEAN_KEYS = (
    "initial_ari_to_generating",
    "final_centroid_displacement",
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
)


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (int(row["n"]), int(row["kmeans_k"]), row["initialization"]), []
        ).append(row)
    aggregates = []
    for (n, kmeans_k, initialization), items in sorted(grouped.items()):
        errors = np.asarray([item["error"] for item in items], dtype=float)
        true_values = np.asarray([item["true_value"] for item in items], dtype=float)
        aggregate = {
            "n": n,
            "kmeans_k": kmeans_k,
            "initialization": initialization,
            "initialization_type": items[0]["initialization_type"],
            "selected_k_best": int(items[0]["selected_k_best"]),
            "k_role": items[0]["k_role"],
            "estimator": "OffCEM K-means",
            "mse": float(np.mean(errors**2)),
            "rel_mse": float(np.mean(errors**2 / np.maximum(true_values**2, 1e-12))),
            "bias2": float(np.mean(errors) ** 2),
            "variance": float(np.mean((errors - errors.mean()) ** 2)),
            "estimate_mean": float(np.mean([item["estimate"] for item in items])),
            "true_value_mean": float(true_values.mean()),
            "n_worlds": len({item["world_seed"] for item in items}),
            "n_rows": len(items),
        }
        for key in AGGREGATED_MEAN_KEYS:
            aggregate[f"{key}_mean"] = float(np.nanmean([item[key] for item in items]))
        aggregates.append(aggregate)
    return aggregates


def load_tidy_csv(path: Path) -> List[Dict]:
    integer_keys = {
        "n", "world_seed", "stream_seed", "stream_index", "model_seed",
        "kmeans_k", "selected_k_best",
    }
    string_keys = {"estimator", "initialization", "initialization_type", "k_role"}
    rows = []
    with open(path) as file:
        for raw in csv.DictReader(file):
            row = {}
            for key, value in raw.items():
                if key in string_keys:
                    row[key] = value
                elif key in integer_keys:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def plot_aggregates(aggregates: List[Dict], out_dir: Path) -> None:
    """Plot fixed K=50 and selected K_best for both initialization conditions."""
    figure, axis = plt.subplots(figsize=(8, 5))
    styles = {
        STANDARD: ("#7f7f7f", "o", "standard K-means"),
        ORACLE_INITIALIZED: ("#2ca02c", "s", "oracle-initialized K-means"),
    }
    for initialization, (color, marker, label) in styles.items():
        selected = [row for row in aggregates if row["initialization"] == initialization]
        fixed = sorted([row for row in selected if row["kmeans_k"] == 50], key=lambda row: row["n"])
        best = sorted(
            [row for row in selected if row["k_role"] != "fixed_50"],
            key=lambda row: row["n"],
        )
        axis.plot(
            [row["n"] for row in fixed],
            [row["rel_mse"] for row in fixed],
            marker=marker,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{label}, K=50",
        )
        axis.plot(
            [row["n"] for row in best],
            [row["rel_mse"] for row in best],
            marker=marker,
            color=color,
            linewidth=1.8,
            label=f"{label}, selected K",
        )
    axis.set_xscale("log")
    axis.set_xlabel("logged interactions n")
    axis.set_ylabel("Relative MSE")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=7, loc="lower left")
    figure.tight_layout()
    figure.savefig(out_dir / "rel_mse_vs_n.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_experiment(args) -> None:
    if args.quick:
        args.n_list = "1000,10000"
        args.k_best_grid = "1000:30;10000:150"
        args.n_worlds = 1
        args.n_streams = 1
        print("[quick] n-list=1000,10000 n-worlds=1")
    n_list = parse_n_list(args.n_list)
    k_best_by_n = parse_k_best_grid(
        args.k_best_grid,
        n_list=n_list,
        n_actions=DEFAULTS["N_ACTIONS"],
    )
    if args.n_worlds <= 0 or args.n_streams <= 0:
        raise ValueError("--n-worlds and --n-streams must be positive")
    if args.generating_n_clusters != 50:
        raise ValueError("this diagnostic currently requires --generating-n-clusters 50")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "kmeans_oracle_initialization_tidy.csv"
    aggregate_path = out_dir / "kmeans_oracle_initialization_aggregate.csv"
    if args.plot_only:
        aggregates = aggregate_results(load_tidy_csv(tidy_path))
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir)
        return

    total_model_tasks = (
        args.n_worlds
        * args.n_streams
        * sum(2 * len(k_cases_for_n(k_best_by_n[n])) for n in n_list)
    )
    all_rows = []
    started = time()
    progress = tqdm(total=total_model_tasks, desc="oracle-initialized K-means models")
    for world_index in range(args.n_worlds):
        progress.set_postfix_str(f"world={world_index + 1} initializing")

        def checkpoint(world_rows):
            with open(out_dir / "latest_rows.json", "w") as file:
                json.dump(all_rows + world_rows, file, indent=2)

        all_rows.extend(
            run_world(
                world_index=world_index,
                n_list=n_list,
                k_best_by_n=k_best_by_n,
                generating_n_clusters=args.generating_n_clusters,
                eps=args.eps,
                reward_std=args.reward_std,
                n_streams=args.n_streams,
                world_seed_base=args.world_seed_base,
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
