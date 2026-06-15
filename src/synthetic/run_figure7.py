"""Run Figure 7 experiment: varying n_clusters with the paper's exact code.

Usage (from the offcem conda env):
    conda activate offcem
    cd /Users/cispa/Documents/OffCEM/icml2023-offcem/src/synthetic
    python run_figure7.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from time import time

from dataset import SyntheticBanditDataset
import numpy as np
from obp.dataset import linear_reward_function
from ope import run_ope, train_reward_model_via_two_stage
import pandas as pd
from pandas import DataFrame
from policy import gen_eps_greedy
from tqdm import tqdm
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config (from conf/setting/default.yaml) ──
N_SEEDS = 300
DIM_CONTEXT = 10
N_VAL_DATA = 3000
N_TEST_DATA = 100000
N_TEST_USERS = 1000
N_ACTIONS = 1000
N_CAT_PER_DIM = 5
LATENT_PARAM_MAT_DIM = 5
N_CAT_DIM = 10
BETA = -0.1
N_DEF_ACTIONS = 0.0
REWARD_STD = 3.0
REWARD_TYPE = "continuous"
EPS = 0.2
RANDOM_STATE = 12345
N_CLUSTERS_LIST = [10, 30, 50, 100, 200]

OUT_DIR = Path("/Users/cispa/Documents/OffCEM/figure7_results")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def run():
    start_time = time()

    all_results = {}
    for n_clusters in N_CLUSTERS_LIST:
        ckpt_path = OUT_DIR / f"results_nc{n_clusters}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                saved = json.load(f)
            all_results[n_clusters] = saved
            print(f"|C|={n_clusters}: loaded {len(saved['IPS'])} seeds from checkpoint")
            continue

        dataset = SyntheticBanditDataset(
            n_actions=N_ACTIONS,
            dim_context=DIM_CONTEXT,
            beta=BETA,
            reward_type=REWARD_TYPE,
            n_cat_per_dim=N_CAT_PER_DIM,
            latent_param_mat_dim=LATENT_PARAM_MAT_DIM,
            n_cat_dim=N_CAT_DIM,
            n_deficient_actions=int(N_ACTIONS * N_DEF_ACTIONS),
            reward_function=linear_reward_function,
            reward_std=REWARD_STD,
            random_state=RANDOM_STATE,
        )

        test_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=N_TEST_DATA, n_users=N_TEST_USERS,
        )
        policy_value = np.average(
            test_bandit_data["expected_reward"],
            weights=gen_eps_greedy(
                expected_reward=test_bandit_data["expected_reward"], eps=EPS,
            )[:, :, 0],
            axis=1,
        ).mean()

        estimated_policy_value_list = []
        t_nc = time()
        for seed_idx in tqdm(range(N_SEEDS), desc=f"|C|={n_clusters}"):
            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=N_VAL_DATA, n_clusters=n_clusters,
            )
            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"], eps=EPS,
            )
            f_x_a, q_x_a = train_reward_model_via_two_stage(
                bandit_data, bandit_data["clusters"],
                random_state=RANDOM_STATE + seed_idx,
            )
            estimated_policy_values = run_ope(
                bandit_data=bandit_data,
                pi_e=pi_e,
                action_clusters=bandit_data["clusters"],
                f_x_a=f_x_a,
                q_x_a=q_x_a,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        # Save results for this n_clusters
        results_dict = {}
        for est_name in estimated_policy_value_list[0].keys():
            results_dict[est_name] = [d[est_name] for d in estimated_policy_value_list]
        results_dict["policy_value"] = float(policy_value)

        with open(ckpt_path, "w") as f:
            json.dump(results_dict, f)

        all_results[n_clusters] = results_dict
        elapsed = (time() - t_nc) / 60
        print(f"|C|={n_clusters}: done in {elapsed:.1f} min  (V(pi)={policy_value:.4f})")

    # ── Compute metrics and plot ──
    print(f"\nTotal time: {(time()-start_time)/60:.1f} min")
    compute_and_plot(all_results)


def compute_and_plot(all_results):
    est_names_paper = ["IPS", "DR", "DM", "MIPS", "OffCEM (true clus + 2s reg)"]
    display_names = {
        "IPS": "IPS",
        "DR": "DR",
        "DM": "DM",
        "MIPS": "MIPS",
        "OffCEM (true clus + 2s reg)": "OffCEM",
    }
    colors = {
        "IPS": "#d62728", "DR": "#ff7f0e", "DM": "#2ca02c",
        "MIPS": "#9467bd", "OffCEM (true clus + 2s reg)": "#1f77b4",
    }
    markers = {
        "IPS": "s", "DR": "^", "DM": "D",
        "MIPS": "v", "OffCEM (true clus + 2s reg)": "o",
    }

    print("\n" + "=" * 80)
    print(f"{'|C|':>5} {'Estimator':>30} {'relMSE':>10} {'relBias^2':>10} {'relVar':>10}")
    print("-" * 80)

    metrics = {}
    for nc in N_CLUSTERS_LIST:
        res = all_results[nc]
        V_true = res["policy_value"]
        norm = V_true ** 2
        metrics[nc] = {}
        for est in est_names_paper:
            v = np.array(res[est])
            bias = v.mean() - V_true
            var = v.var(ddof=0)
            mse = bias**2 + var
            metrics[nc][est] = dict(
                relMSE=mse / norm, relBias2=bias**2 / norm, relVar=var / norm,
            )
            print(f"{nc:5d} {est:>30s} {mse/norm:10.4f} {bias**2/norm:10.4f} {var/norm:10.4f}")
        print("-" * 80)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = ["relMSE", "relBias2", "relVar"]
    titles = ["Normalized MSE", "Relative Squared Bias", "Relative Sample Variance"]

    for ax, key, title in zip(axes, keys, titles):
        for est in est_names_paper:
            vals = [metrics[nc][est][key] for nc in N_CLUSTERS_LIST]
            ax.plot(
                N_CLUSTERS_LIST, vals,
                marker=markers[est], color=colors[est],
                label=display_names[est], linewidth=2, markersize=7,
            )
        ax.set_xlabel("number of clusters", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(N_CLUSTERS_LIST)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Figure 7 (Paper Code): Varying |C| with |A|=1000, n=3000, "
        + "ε=0.2, σ=3.0",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    save_path = "/Users/cispa/Documents/OffCEM/figure7_paper_code.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    run()
