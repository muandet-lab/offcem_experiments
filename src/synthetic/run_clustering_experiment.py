"""Clustering experiment: vary method x n_clusters x balance.

Usage:
    cd /Users/cispa/Documents/OffCEM/icml2023-offcem-expts/src/synthetic
    python run_clustering_experiment.py --mode matched --quick
    python run_clustering_experiment.py --mode mismatched --n-seeds 50
    python run_clustering_experiment.py --mode both
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
from dataset import SyntheticBanditDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import linear_reward_function
from obp.ope import RegressionModel
from ope import run_ope
from ope import run_ope_with_clustering_variants
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy
from sklearn.neural_network import MLPRegressor as MLP
from tqdm import tqdm

# ── Defaults matching the paper ──
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
    p = argparse.ArgumentParser(description="OffCEM clustering experiments")
    p.add_argument(
        "--mode",
        choices=["matched", "mismatched", "both"],
        default="matched",
    )
    p.add_argument("--n-seeds", type=int, default=300)
    p.add_argument(
        "--n-clusters",
        type=str,
        default="10,20,30,40,50,75,100,150,200",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="kmeans,spectral,agglomerative,random,original",
    )
    p.add_argument(
        "--balances",
        type=str,
        default="natural,balanced,power_law",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/clustering_results",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 5 seeds, 2 n_clusters values",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip summary plot generation. Useful for parallel/array workers.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate summary plots from existing checkpoints.",
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


def compute_policy_value(dataset, gen_method, gen_balance, n_clusters):
    C = DEFAULTS
    test_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=C["N_TEST_DATA"],
        n_users=C["N_TEST_USERS"],
        n_clusters=n_clusters,
        clustering_method=gen_method,
        cluster_balance=gen_balance,
    )
    return np.average(
        test_data["expected_reward"],
        weights=gen_eps_greedy(
            expected_reward=test_data["expected_reward"],
            eps=C["EPS"],
        )[:, :, 0],
        axis=1,
    ).mean()


# ── Matched mode ──


def run_matched(args):
    """For each (method, n_clusters, balance) triple:
    generate data AND estimate with the same clustering.
    """
    out_dir = Path(args.out_dir) / "matched"
    out_dir.mkdir(exist_ok=True, parents=True)
    C = DEFAULTS

    methods = args.methods.split(",")
    n_clusters_list = [int(x) for x in args.n_clusters.split(",")]
    balances = args.balances.split(",")

    total = len(balances) * len(n_clusters_list) * len(methods)
    done = 0

    for balance in balances:
        for n_clusters in n_clusters_list:
            for method in methods:
                combo_key = f"{method}_nc{n_clusters}_{balance}"
                ckpt_path = out_dir / f"results_{combo_key}.json"

                if ckpt_path.exists():
                    done += 1
                    print(f"[matched] {combo_key}: loaded from checkpoint ({done}/{total})")
                    continue

                done += 1
                print(f"[matched] Running {combo_key} ... ({done}/{total})")
                t0 = time()

                dataset = make_dataset()
                policy_value = compute_policy_value(
                    dataset,
                    method,
                    balance,
                    n_clusters,
                )

                estimated_policy_value_list = []
                for seed_idx in tqdm(range(args.n_seeds), desc=combo_key):
                    bandit_data = dataset.obtain_batch_bandit_feedback(
                        n_rounds=C["N_VAL_DATA"],
                        n_clusters=n_clusters,
                        clustering_method=method,
                        cluster_balance=balance,
                    )
                    pi_e = gen_eps_greedy(
                        expected_reward=bandit_data["expected_reward"],
                        eps=C["EPS"],
                    )
                    f_x_a, q_x_a = train_reward_model_via_two_stage(
                        bandit_data,
                        bandit_data["clusters"],
                        random_state=C["RANDOM_STATE"] + seed_idx,
                    )
                    est_values = run_ope(
                        bandit_data=bandit_data,
                        pi_e=pi_e,
                        action_clusters=bandit_data["clusters"],
                        f_x_a=f_x_a,
                        q_x_a=q_x_a,
                    )
                    estimated_policy_value_list.append(est_values)

                results = {}
                for est_name in estimated_policy_value_list[0].keys():
                    results[est_name] = [
                        d[est_name] for d in estimated_policy_value_list
                    ]
                results["policy_value"] = float(policy_value)

                with open(ckpt_path, "w") as f:
                    json.dump(results, f)

                elapsed = (time() - t0) / 60
                print(f"  -> {combo_key}: done in {elapsed:.1f} min")


# ── Mismatched mode ──


def run_mismatched(args):
    """Generate data with 'original'/natural clustering.
    Then run OPE with every estimation method.
    """
    out_dir = Path(args.out_dir) / "mismatched"
    out_dir.mkdir(exist_ok=True, parents=True)
    C = DEFAULTS

    methods = args.methods.split(",")
    n_clusters_list = [int(x) for x in args.n_clusters.split(",")]

    for n_clusters in n_clusters_list:
        ckpt_path = out_dir / f"results_nc{n_clusters}.json"
        if ckpt_path.exists():
            print(f"[mismatched] nc={n_clusters}: loaded from checkpoint")
            continue

        print(f"[mismatched] nc={n_clusters}: running with {len(methods)} est methods ...")
        t0 = time()

        dataset = make_dataset()
        policy_value = compute_policy_value(
            dataset,
            "original",
            "natural",
            n_clusters,
        )

        all_estimated = []
        for seed_idx in tqdm(range(args.n_seeds), desc=f"mismatched nc={n_clusters}"):
            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=C["N_VAL_DATA"],
                n_clusters=n_clusters,
                clustering_method="original",
                cluster_balance="natural",
            )
            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"],
                eps=C["EPS"],
            )

            est_clusters_dict = {}
            f_x_a_dict = {}
            for m in methods:
                if m == "original":
                    est_c3d = bandit_data["clusters"]
                else:
                    est_c1d = compute_clusters(
                        bandit_data["action_context_one_hot"],
                        n_clusters,
                        method=m,
                        balance="natural",
                        random_state=C["RANDOM_STATE"],
                    )
                    est_c3d = clusters_to_onehot_3d(est_c1d, bandit_data["n_users"])
                est_clusters_dict[m] = est_c3d

                f_x_a = train_reward_model_via_two_stage(
                    bandit_data,
                    est_c3d,
                    need_q_x_a=False,
                    random_state=C["RANDOM_STATE"] + seed_idx,
                )
                f_x_a_dict[m] = f_x_a

            reg_model = RegressionModel(
                n_actions=bandit_data["n_actions"],
                action_context=bandit_data["action_context_one_hot"],
                base_model=MLP(
                    hidden_layer_sizes=(50, 50, 50),
                    random_state=C["RANDOM_STATE"] + seed_idx,
                ),
            )
            q_x_a = reg_model.fit_predict(
                context=bandit_data["context"],
                action=bandit_data["action"],
                reward=bandit_data["reward"],
            )

            results = run_ope_with_clustering_variants(
                bandit_data=bandit_data,
                pi_e=pi_e,
                estimation_clusters_dict=est_clusters_dict,
                f_x_a_dict=f_x_a_dict,
                q_x_a=q_x_a,
            )
            all_estimated.append(results)

        output = {}
        for est_name in all_estimated[0].keys():
            output[est_name] = [d[est_name] for d in all_estimated]
        output["policy_value"] = float(policy_value)

        with open(ckpt_path, "w") as f:
            json.dump(output, f)

        elapsed = (time() - t0) / 60
        print(f"  -> nc={n_clusters}: done in {elapsed:.1f} min")


# ── Plotting ──


def compute_metrics(results, est_names):
    V_true = results["policy_value"]
    norm = V_true ** 2
    metrics = {}
    for est in est_names:
        if est not in results:
            continue
        v = np.array(results[est])
        bias = v.mean() - V_true
        var = v.var(ddof=0)
        mse = bias ** 2 + var
        metrics[est] = dict(
            relMSE=mse / norm,
            relBias2=bias ** 2 / norm,
            relVar=var / norm,
        )
    return metrics


def plot_matched_results(out_dir):
    out_dir = Path(out_dir) / "matched"
    if not out_dir.exists():
        print("No matched results to plot.")
        return

    results_files = sorted(out_dir.glob("results_*.json"))
    if not results_files:
        print("No matched result files found.")
        return

    all_data = {}
    for f in results_files:
        key = f.stem.replace("results_", "")
        with open(f) as fh:
            all_data[key] = json.load(fh)

    combos = {}
    for key in all_data:
        parts = key.rsplit("_", 2)
        method = parts[0]
        nc = int(parts[1].replace("nc", ""))
        balance = parts[2]
        combos.setdefault(balance, {}).setdefault(method, {})[nc] = all_data[key]

    offcem_2s = "OffCEM (true clus + 2s reg)"
    colors = {
        "original": "#1f77b4",
        "kmeans": "#ff7f0e",
        "spectral": "#2ca02c",
        "agglomerative": "#d62728",
        "random": "#9467bd",
    }
    markers = {
        "original": "o",
        "kmeans": "s",
        "spectral": "^",
        "agglomerative": "D",
        "random": "v",
    }

    for balance, methods_data in combos.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        metric_keys = ["relMSE", "relBias2", "relVar"]
        titles = ["Normalized MSE", "Relative Squared Bias", "Relative Sample Variance"]

        for ax, mkey, title in zip(axes, metric_keys, titles):
            for method, nc_data in sorted(methods_data.items()):
                nc_list = sorted(nc_data.keys())
                vals = []
                for nc in nc_list:
                    m = compute_metrics(nc_data[nc], [offcem_2s])
                    vals.append(m.get(offcem_2s, {}).get(mkey, float("nan")))
                ax.plot(
                    nc_list,
                    vals,
                    marker=markers.get(method, "x"),
                    color=colors.get(method, "gray"),
                    label=method,
                    linewidth=2,
                    markersize=7,
                )
            ax.set_xlabel("number of clusters", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        fig.suptitle(
            f"Matched Clustering ({balance} balance): OffCEM 2-stage",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()
        save_path = Path(out_dir).parent / f"matched_{balance}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")


def plot_mismatched_results(out_dir):
    out_dir = Path(out_dir) / "mismatched"
    if not out_dir.exists():
        print("No mismatched results to plot.")
        return

    results_files = sorted(out_dir.glob("results_nc*.json"))
    if not results_files:
        print("No mismatched result files found.")
        return

    nc_results = {}
    for f in results_files:
        nc = int(f.stem.replace("results_nc", ""))
        with open(f) as fh:
            nc_results[nc] = json.load(fh)

    nc_list = sorted(nc_results.keys())

    est_names = [
        k for k in nc_results[nc_list[0]].keys()
        if k.startswith("OffCEM (") and k != "policy_value"
    ]

    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers_list = ["o", "s", "^", "D", "v", "x"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_keys = ["relMSE", "relBias2", "relVar"]
    titles = ["Normalized MSE", "Relative Squared Bias", "Relative Sample Variance"]

    for ax, mkey, title in zip(axes, metric_keys, titles):
        for i, est in enumerate(est_names):
            vals = []
            for nc in nc_list:
                m = compute_metrics(nc_results[nc], [est])
                vals.append(m.get(est, {}).get(mkey, float("nan")))
            label = est.replace("OffCEM (", "").rstrip(")")
            ax.plot(
                nc_list,
                vals,
                marker=markers_list[i % len(markers_list)],
                color=colors_list[i % len(colors_list)],
                label=label,
                linewidth=2,
                markersize=7,
            )
        ax.set_xlabel("number of clusters", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Mismatched: Data=original, Estimation=varied",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    save_path = Path(out_dir).parent / "mismatched_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_all(out_dir):
    plot_matched_results(out_dir)
    plot_mismatched_results(out_dir)


# ── Main ──


def main():
    args = parse_args()

    if args.plot_only:
        plot_all(args.out_dir)
        return

    if args.quick:
        args.n_seeds = 5
        args.n_clusters = "10,50"
        print(f"[quick mode] n_seeds={args.n_seeds}, n_clusters={args.n_clusters}")

    start_time = time()

    if args.mode in ("matched", "both"):
        run_matched(args)
    if args.mode in ("mismatched", "both"):
        run_mismatched(args)

    print(f"\nTotal time: {(time()-start_time)/60:.1f} min")
    if not args.no_plot:
        plot_all(args.out_dir)


if __name__ == "__main__":
    main()
