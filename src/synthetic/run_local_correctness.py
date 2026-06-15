"""Local correctness analysis for OffCEM under random clustering.

Phase 1: Fix n_clusters, vary seeds. Check whether OffCEM's within-cluster
reward regression holds when clusters have no feature/reward coherence.

Usage:
    cd /Users/cispa/Documents/OffCEM/icml2023-offcem-expts/src/synthetic
    python run_local_correctness.py --n-clusters 50 --n-seeds 10
    python run_local_correctness.py --n-clusters 50 --n-seeds 10 --methods random,shuffled_partition
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from time import time

from clustering import ClusterMethod
from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from clustering import corrupt_partition
from dataset import SyntheticBanditDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import linear_reward_function
from obp.ope import RegressionModel
from ope import run_ope
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy
from sklearn.neural_network import MLPRegressor as MLP
from tqdm import tqdm

DEFAULTS = dict(
    DIM_CONTEXT=10,
    N_VAL_DATA=3000,
    N_TEST_DATA=100000,
    N_TEST_USERS=1000,
    N_ACTIONS=1000,
    N_CAT_PER_DIM=5,
    LATENT_PARAM_MAT_DIM=5,
    N_CAT_DIM=10,
    BETA=-0.1,
    N_DEF_ACTIONS=0.0,
    REWARD_STD=3.0,
    REWARD_TYPE="continuous",
    EPS=0.2,
    RANDOM_STATE=12345,
)


def parse_args():
    p = argparse.ArgumentParser(description="Local correctness analysis")
    p.add_argument("--n-clusters", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument(
        "--methods",
        type=str,
        default="matched,original,random,shuffled_partition",
        help=(
            "Comma-separated estimation clusterings. Controls: "
            "'matched' reuses the dataset's exact generating partition "
            "(positive control, local correctness should hold); "
            "'original' re-runs the paper's method -> a DIFFERENT partition "
            "(same-method/different-partition isolation control); "
            "'random'/'shuffled_partition' are negative controls."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/local_correctness_results",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Softmax temperature for 'original' clustering (default: 10.0, lower = sharper)",
    )
    p.add_argument(
        "--gen-clustering-method",
        type=str,
        default="original",
        help=(
            "Clustering method used to GENERATE the dataset's reward-defining "
            "partition (default: 'original', the paper's non-geometric stochastic "
            "softmax). Set to 'feature_bucket' or 'kmeans' for the FEATURE-COHERENT "
            "benchmark: the partition becomes a deterministic function of action "
            "features, so the per-cluster reward genuinely tracks features. Then "
            "test whether feature-based estimation (--methods kmeans/feature_bucket) "
            "recovers matched-like local correctness. Use a distinct --out-dir so "
            "these results stay separate from the default-benchmark results."
        ),
    )
    p.add_argument(
        "--corrupt-p",
        type=float,
        default=0.0,
        help=(
            "For the 'corrupt' method: fraction of actions reassigned from the "
            "true partition to random clusters. 0.0 = exact partition (==matched), "
            "1.0 = fully random. Sweep this with separate --out-dir per value."
        ),
    )
    return p.parse_args()


def make_dataset():
    C = DEFAULTS
    return SyntheticBanditDataset(
        n_actions=C["N_ACTIONS"],
        dim_context=C["DIM_CONTEXT"],
        beta=C["BETA"],
        reward_type=C["REWARD_TYPE"],
        n_cat_per_dim=C["N_CAT_PER_DIM"],
        latent_param_mat_dim=C["LATENT_PARAM_MAT_DIM"],
        n_cat_dim=C["N_CAT_DIM"],
        n_deficient_actions=int(C["N_ACTIONS"] * C["N_DEF_ACTIONS"]),
        reward_function=linear_reward_function,
        reward_std=C["REWARD_STD"],
        random_state=C["RANDOM_STATE"],
    )


def analyze_local_correctness(
    bandit_data,
    clusters_1d,
    f_x_a,
    q_x_a,
    seed_idx,
):
    """Compute per-cluster regression error statistics.

    Returns a dict with per-cluster and aggregate error metrics for both
    the 2-stage (f_x_a) and 1-stage (q_x_a) reward models.
    """
    expected_reward = bandit_data["expected_reward"]
    n_rounds = expected_reward.shape[0]
    n_actions = expected_reward.shape[1]
    n_clusters = int(clusters_1d.max()) + 1

    f_x_a_2d = f_x_a[:, :, 0] if f_x_a.ndim == 3 else f_x_a
    q_x_a_2d = q_x_a[:, :, 0] if q_x_a.ndim == 3 else q_x_a

    err_2s = f_x_a_2d - expected_reward
    err_1s = q_x_a_2d - expected_reward

    # Within-cluster DEMEANED error: for each context (row) and each cluster,
    # subtract the mean error over actions in that cluster. This isolates the
    # quantity OffCEM's local-correctness condition actually concerns -- the
    # RELATIVE within-cluster reward differences. The removed per-cluster
    # baseline is absorbed by OffCEM's cluster-marginal importance weighting,
    # so it must NOT count against local correctness. Absolute err_*_mse below
    # conflates this absorbable baseline with the real local-correctness error.
    err_2s_dm = err_2s.copy()
    err_1s_dm = err_1s.copy()
    for c in range(n_clusters):
        mask = clusters_1d == c
        if not mask.any():
            continue
        err_2s_dm[:, mask] -= err_2s[:, mask].mean(axis=1, keepdims=True)
        err_1s_dm[:, mask] -= err_1s[:, mask].mean(axis=1, keepdims=True)

    cluster_stats = []
    for c in range(n_clusters):
        mask = clusters_1d == c
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue

        err_2s_c = err_2s[:, mask].ravel()
        err_1s_c = err_1s[:, mask].ravel()
        err_2s_dm_c = err_2s_dm[:, mask].ravel()
        err_1s_dm_c = err_1s_dm[:, mask].ravel()

        cluster_stats.append(dict(
            cluster_id=c,
            cluster_size=cluster_size,
            err_2s_mean=float(err_2s_c.mean()),
            err_2s_std=float(err_2s_c.std()),
            err_2s_abs_mean=float(np.abs(err_2s_c).mean()),
            err_2s_mse=float((err_2s_c ** 2).mean()),
            err_2s_dm_mse=float((err_2s_dm_c ** 2).mean()),
            err_2s_median=float(np.median(err_2s_c)),
            err_2s_q10=float(np.percentile(err_2s_c, 10)),
            err_2s_q90=float(np.percentile(err_2s_c, 90)),
            err_1s_mean=float(err_1s_c.mean()),
            err_1s_std=float(err_1s_c.std()),
            err_1s_abs_mean=float(np.abs(err_1s_c).mean()),
            err_1s_mse=float((err_1s_c ** 2).mean()),
            err_1s_dm_mse=float((err_1s_dm_c ** 2).mean()),
            err_1s_median=float(np.median(err_1s_c)),
            err_1s_q10=float(np.percentile(err_1s_c, 10)),
            err_1s_q90=float(np.percentile(err_1s_c, 90)),
        ))

    return dict(
        seed=seed_idx,
        n_clusters=n_clusters,
        cluster_stats=cluster_stats,
        agg_2s_mse=float((err_2s ** 2).mean()),
        agg_2s_bias=float(err_2s.mean()),
        agg_2s_abs_mean=float(np.abs(err_2s).mean()),
        # local-correctness-relevant (within-cluster relative) error
        agg_2s_dm_mse=float((err_2s_dm ** 2).mean()),
        agg_1s_mse=float((err_1s ** 2).mean()),
        agg_1s_bias=float(err_1s.mean()),
        agg_1s_abs_mean=float(np.abs(err_1s).mean()),
        agg_1s_dm_mse=float((err_1s_dm ** 2).mean()),
        err_2s_all=err_2s.ravel().tolist(),
        err_1s_all=err_1s.ravel().tolist(),
    )


def run_analysis(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    C = DEFAULTS

    methods = args.methods.split(",")
    dataset = make_dataset()

    print(f"Action set size: {C['N_ACTIONS']}")
    print(f"Cluster size: {C['N_ACTIONS'] // args.n_clusters} actions/cluster")

    for method in methods:
        method_dir = out_dir / method
        method_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Method: {method}, n_clusters={args.n_clusters}")
        print(f"{'='*60}")

        all_results = []
        for seed_idx in tqdm(range(args.n_seeds), desc=method):
            cluster_seed = C["RANDOM_STATE"] + seed_idx

            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=C["N_VAL_DATA"],
                n_clusters=args.n_clusters,
                clustering_method=args.gen_clustering_method,
                cluster_balance="natural",
                cluster_temperature=args.temperature,
            )

            if method == "matched":
                # Positive control: reuse the EXACT partition the dataset used to
                # construct rewards, taken straight from the dataset. This is the
                # correct ground-truth partition for any --gen-clustering-method.
                # (It is also the only way to recover 'original', which is
                # non-reproducible: sample_action_fast falls back to the global RNG
                # and the seed is never forwarded.)
                clusters_1d = bandit_data["cluster_indices"]
                clusters_3d = bandit_data["clusters"]
            elif method == "corrupt":
                # Graded knob: start from the true partition, reassign a fraction
                # args.corrupt_p of actions at random (p=0 == matched, p=1 ~ random).
                clusters_1d = corrupt_partition(
                    bandit_data["cluster_indices"], args.corrupt_p, cluster_seed
                )
                clusters_3d = clusters_to_onehot_3d(clusters_1d, bandit_data["n_users"])
            else:
                clusters_1d = compute_clusters(
                    bandit_data["action_context_one_hot"],
                    args.n_clusters,
                    method=method,
                    balance="natural",
                    random_state=cluster_seed,
                )
                clusters_3d = clusters_to_onehot_3d(clusters_1d, bandit_data["n_users"])

            f_x_a, q_x_a = train_reward_model_via_two_stage(
                bandit_data,
                clusters_3d,
                random_state=cluster_seed,
            )

            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"],
                eps=C["EPS"],
            )
            est_values = run_ope(
                bandit_data=bandit_data,
                pi_e=pi_e,
                action_clusters=clusters_3d,
                f_x_a=f_x_a,
                q_x_a=q_x_a,
            )

            result = analyze_local_correctness(
                bandit_data, clusters_1d, f_x_a, q_x_a, seed_idx,
            )
            result["estimator_values"] = est_values
            result["cluster_seed"] = cluster_seed
            all_results.append(result)

        save_path = method_dir / f"nc{args.n_clusters}.json"
        serializable = []
        all_err_2s = []
        all_err_1s = []
        for r in all_results:
            all_err_2s.extend(r.pop("err_2s_all"))
            all_err_1s.extend(r.pop("err_1s_all"))
            serializable.append(r)

        with open(save_path, "w") as f:
            json.dump(dict(
                method=method,
                n_clusters=args.n_clusters,
                n_seeds=args.n_seeds,
                per_seed=serializable,
            ), f, indent=2)

        print(f"Results saved to {save_path}")

        # Headline summary (mean over seeds). The DEMEANED rows are the
        # local-correctness-relevant metric; the absolute rows are the old metric.
        def _mean(key):
            return float(np.mean([r[key] for r in all_results]))
        offcem_keys = [k for k in all_results[0]["estimator_values"] if "offcem" in k.lower()]
        print(f"  [{method}] reward-model MSE (mean over {args.n_seeds} seeds):")
        print(f"    absolute : 2-stage={_mean('agg_2s_mse'):7.3f}  1-stage={_mean('agg_1s_mse'):7.3f}")
        print(f"    demeaned : 2-stage={_mean('agg_2s_dm_mse'):7.3f}  1-stage={_mean('agg_1s_dm_mse'):7.3f}  <- local correctness")
        for ek in offcem_keys:
            ev = float(np.mean([r["estimator_values"][ek] for r in all_results]))
            print(f"    est[{ek}]={ev:7.3f}")

        plot_results(method, args.n_clusters, all_results,
                     np.array(all_err_2s), np.array(all_err_1s), method_dir)


def plot_results(method, n_clusters, results, err_2s_all, err_1s_all, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Error histograms (2-stage vs 1-stage)
    ax = axes[0, 0]
    clip = np.percentile(np.abs(err_2s_all), 99)
    ax.hist(err_2s_all, bins=100, range=(-clip, clip), alpha=0.7, color="#1f77b4", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("2-Stage Reward Error Distribution", fontweight="bold")
    ax.set_xlabel("f(x,a) - Q(x,a)")
    ax.set_ylabel("density")

    ax = axes[0, 1]
    clip = np.percentile(np.abs(err_1s_all), 99)
    ax.hist(err_1s_all, bins=100, range=(-clip, clip), alpha=0.7, color="#ff7f0e", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("1-Stage Reward Error Distribution", fontweight="bold")
    ax.set_xlabel("q(x,a) - Q(x,a)")

    # Row 1 col 3: QQ-style: per-cluster MSE distribution across seeds
    ax = axes[0, 2]
    per_cluster_mse_2s = []
    per_cluster_mse_1s = []
    for r in results:
        for cs in r["cluster_stats"]:
            per_cluster_mse_2s.append(cs["err_2s_mse"])
            per_cluster_mse_1s.append(cs["err_1s_mse"])
    per_cluster_mse_2s = np.array(per_cluster_mse_2s)
    per_cluster_mse_1s = np.array(per_cluster_mse_1s)
    ax.hist(per_cluster_mse_2s, bins=50, alpha=0.6, color="#1f77b4", density=True, label="2-stage")
    ax.hist(per_cluster_mse_1s, bins=50, alpha=0.6, color="#ff7f0e", density=True, label="1-stage")
    ax.set_title("Per-Cluster MSE Distribution", fontweight="bold")
    ax.set_xlabel("within-cluster MSE")
    ax.legend()

    # Row 2 col 1: Per-seed aggregate MSE comparison
    ax = axes[1, 0]
    seeds = [r["seed"] for r in results]
    mse_2s = [r["agg_2s_mse"] for r in results]
    mse_1s = [r["agg_1s_mse"] for r in results]
    ax.bar(np.array(seeds) - 0.15, mse_2s, 0.3, label="2-stage", color="#1f77b4")
    ax.bar(np.array(seeds) + 0.15, mse_1s, 0.3, label="1-stage", color="#ff7f0e")
    ax.set_title("Aggregate Reward MSE by Seed", fontweight="bold")
    ax.set_xlabel("seed")
    ax.set_ylabel("MSE")
    ax.legend()

    # Row 2 col 2: Per-seed estimator values
    ax = axes[1, 1]
    est_names = list(results[0]["estimator_values"].keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, est in enumerate(est_names):
        vals = [r["estimator_values"][est] for r in results]
        ax.plot(seeds, vals, marker="o", label=est, color=colors[i % len(colors)], markersize=4)
    ax.set_title("Estimator Values by Seed", fontweight="bold")
    ax.set_xlabel("seed")
    ax.set_ylabel("estimated policy value")
    ax.legend(fontsize=7)

    # Row 2 col 3: Per-cluster bias spread (boxplot of per-cluster mean errors)
    ax = axes[1, 2]
    bias_per_cluster_2s = []
    bias_per_cluster_1s = []
    for r in results:
        for cs in r["cluster_stats"]:
            bias_per_cluster_2s.append(cs["err_2s_mean"])
            bias_per_cluster_1s.append(cs["err_1s_mean"])
    ax.boxplot(
        [bias_per_cluster_2s, bias_per_cluster_1s],
        labels=["2-stage", "1-stage"],
        showfliers=False,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Per-Cluster Bias Distribution", fontweight="bold")
    ax.set_ylabel("mean error within cluster")

    fig.suptitle(
        f"Local Correctness: {method}, n_clusters={n_clusters}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = out_dir / f"local_correctness_nc{n_clusters}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    args = parse_args()
    t0 = time()
    run_analysis(args)
    print(f"\nTotal time: {(time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
