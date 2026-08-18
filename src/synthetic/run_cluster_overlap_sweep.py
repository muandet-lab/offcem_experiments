"""Cluster-level overlap/support stress test for OffCEM.

This experiment keeps local correctness favorable by using the matched
generating partition, then degrades behavior-policy mass on clusters favored by
a fixed target policy. It tests the variance/support axis complementary to the
policy-disagreement sweep.

Run from ``src/synthetic``:

    conda run -n offcem python -m run_cluster_overlap_sweep --quick
"""
import argparse
import csv
import json
import warnings
from pathlib import Path
from time import time
from typing import Dict
from typing import Iterable
from typing import List

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import linear_reward_function
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import RegressionModel
from obp.utils import softmax
from sklearn.neural_network import MLPRegressor as MLP
import torch
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from dataset import SyntheticBanditDataset
from ope import OffCEM
from ope import OffPolicyEvaluation
from ope import train_reward_model_via_two_stage


DEFAULTS = dict(
    DIM_CONTEXT=10,
    N_USERS=200,
    N_ACTIONS=1000,
    N_CAT_PER_DIM=5,
    LATENT_PARAM_MAT_DIM=5,
    N_CAT_DIM=10,
    BETA=-0.1,
    N_DEF_ACTIONS=0.0,
    REWARD_TYPE="continuous",
    RANDOM_STATE=12345,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster-level overlap/support stress test"
    )
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-actions", type=int, default=DEFAULTS["N_ACTIONS"])
    parser.add_argument("--n-users", type=int, default=DEFAULTS["N_USERS"])
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=DEFAULTS["BETA"])
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--target-strength", type=float, default=1.0)
    parser.add_argument(
        "--overlap-list",
        type=str,
        default="1,.5,.25,.1,.05,.01,0",
        help="Mass multiplier for target-favored clusters under the logger",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/cluster_overlap_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use n_rounds=1000, n_seeds=2, overlap=1,.1",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate CSV and plots from tidy results",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_overlap_list(value: str) -> List[float]:
    levels = [float(item) for item in value.split(",") if item]
    if not levels:
        raise ValueError("--overlap-list must contain at least one value")
    if any(level < 0.0 or level > 1.0 for level in levels):
        raise ValueError("overlap levels must lie in [0, 1]")
    return sorted(set(levels), reverse=True)


def make_dataset(
    seed: int,
    n_actions: int,
    beta: float,
    reward_std: float,
) -> SyntheticBanditDataset:
    return SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=DEFAULTS["DIM_CONTEXT"],
        beta=beta,
        reward_type=DEFAULTS["REWARD_TYPE"],
        n_cat_per_dim=DEFAULTS["N_CAT_PER_DIM"],
        latent_param_mat_dim=DEFAULTS["LATENT_PARAM_MAT_DIM"],
        n_cat_dim=DEFAULTS["N_CAT_DIM"],
        n_deficient_actions=int(n_actions * DEFAULTS["N_DEF_ACTIONS"]),
        reward_function=linear_reward_function,
        reward_std=reward_std,
        random_state=seed,
    )


def reconstruct_pi0_population(
    fixed_expected_rewards: np.ndarray,
    beta: float,
) -> np.ndarray:
    return softmax(beta * fixed_expected_rewards)


def cluster_policy_mass(policy: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=float)
    n_clusters = int(clusters.max()) + 1
    masses = np.zeros((policy.shape[0], n_clusters), dtype=float)
    for c in range(n_clusters):
        masses[:, c] = policy[:, clusters == c].sum(axis=1)
    return masses


def build_cluster_favoring_target(
    q_population: np.ndarray,
    pi0_population: np.ndarray,
    clusters: np.ndarray,
    tau: float,
    strength: float,
) -> np.ndarray:
    """Shift target mass toward high-reward clusters, preserving pi0 within c."""
    if tau <= 0:
        raise ValueError("target tau must be positive")
    if strength < 0.0 or strength > 1.0:
        raise ValueError("target strength must lie in [0, 1]")
    pi0_c = cluster_policy_mass(pi0_population, clusters)
    n_clusters = pi0_c.shape[1]
    cluster_scores = np.zeros_like(pi0_c)
    for c in range(n_clusters):
        cluster_scores[:, c] = q_population[:, clusters == c].mean(axis=1)
    reward_cluster_mass = softmax(cluster_scores / tau)
    target_c = (1.0 - strength) * pi0_c + strength * reward_cluster_mass

    target = np.zeros_like(pi0_population)
    for c in range(n_clusters):
        idx = clusters == c
        pi0_cond = pi0_population[:, idx] / np.maximum(
            pi0_c[:, [c]],
            1e-12,
        )
        target[:, idx] = target_c[:, [c]] * pi0_cond
    assert_policy_matrix(target)
    return target


def degrade_behavior_cluster_overlap(
    pi0_population: np.ndarray,
    target_population: np.ndarray,
    clusters: np.ndarray,
    overlap_level: float,
) -> np.ndarray:
    """Reduce logger mass on clusters whose target mass exceeds pi0 mass."""
    if overlap_level < 0.0 or overlap_level > 1.0:
        raise ValueError("overlap_level must lie in [0, 1]")
    pi0_c = cluster_policy_mass(pi0_population, clusters)
    target_c = cluster_policy_mass(target_population, clusters)
    favored = target_c > pi0_c + 1e-12
    behavior_c = pi0_c.copy()
    behavior_c[favored] *= overlap_level
    behavior_c /= behavior_c.sum(axis=1, keepdims=True)

    behavior = np.zeros_like(pi0_population)
    for c in range(behavior_c.shape[1]):
        idx = clusters == c
        pi0_cond = pi0_population[:, idx] / np.maximum(
            pi0_c[:, [c]],
            1e-12,
        )
        behavior[:, idx] = behavior_c[:, [c]] * pi0_cond
    assert_policy_matrix(behavior)
    return behavior


def assert_policy_matrix(policy: np.ndarray) -> None:
    if policy.ndim != 2:
        raise AssertionError("policy must be 2D")
    if np.any(policy < -1e-12):
        raise AssertionError("policy has negative mass")
    if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("policy rows must sum to one")


def resample_logged_data(
    base_data: Dict,
    behavior_population: np.ndarray,
    reward_std: float,
    sample_seed: int,
) -> Dict:
    """Reuse the fixed world and user stream, but resample actions/rewards."""
    rng = np.random.RandomState(sample_seed)
    data = dict(base_data)
    n_rounds = int(base_data["n_rounds"])
    n_users = int(base_data["n_users"])
    n_actions = int(base_data["n_actions"])
    user_idx = base_data["user_idx"].copy()
    pi_b_rows = behavior_population[user_idx]
    actions = np.array(
        [rng.choice(n_actions, p=probability) for probability in pi_b_rows],
        dtype=int,
    )
    expected_reward = base_data["fixed_expected_rewards"][user_idx]
    factual_mean = expected_reward[np.arange(n_rounds), actions]
    rewards = rng.normal(factual_mean, reward_std, size=n_rounds)

    reward_sum_mat = np.zeros((n_users, n_actions), dtype=float)
    obs_count_mat = np.zeros((n_users, n_actions), dtype=int)
    for user, action, reward in zip(user_idx, actions, rewards):
        reward_sum_mat[int(user), int(action)] += float(reward)
        obs_count_mat[int(user), int(action)] += 1
    reward_mat = np.zeros((n_users, n_actions), dtype=float)
    observed = obs_count_mat > 0
    reward_mat[observed] = reward_sum_mat[observed] / obs_count_mat[observed]

    data.update(
        {
            "user_idx": user_idx,
            "context": base_data["fixed_user_contexts"][user_idx],
            "action": actions,
            "reward": rewards,
            "expected_reward": expected_reward,
            "pi_b": pi_b_rows[:, :, np.newaxis],
            "pscore": pi_b_rows[np.arange(n_rounds), actions],
            "action_embed": base_data["action_context"][actions],
            "reward_sum_mat": reward_sum_mat,
            "obs_count_mat": obs_count_mat,
            "reward_mat": reward_mat,
            "obs_mat": observed.astype(int),
        }
    )
    if int(data["obs_count_mat"].sum()) != n_rounds:
        raise AssertionError("obs_count_mat.sum() must equal n_rounds")
    return data


def cluster_overlap_diagnostics(
    behavior_population: np.ndarray,
    target_population: np.ndarray,
    clusters: np.ndarray,
    user_idx: np.ndarray,
    observed_actions: np.ndarray,
) -> Dict[str, float]:
    pi_b_c = cluster_policy_mass(behavior_population, clusters)
    pi_e_c = cluster_policy_mass(target_population, clusters)
    weights = pi_e_c / np.maximum(pi_b_c, 1e-12)
    observed_clusters = clusters[observed_actions]
    observed_weights = weights[user_idx, observed_clusters]
    unsupported = (pi_b_c <= 1e-12) & (pi_e_c > 1e-12)
    globally_unsupported = (pi_b_c <= 1e-12).all(axis=0) & (
        pi_e_c > 1e-12
    ).any(axis=0)
    contextually_zero = (pi_b_c <= 1e-12).any(axis=0) & (
        pi_e_c > 1e-12
    ).any(axis=0)
    return {
        "cluster_weight_mean": float(observed_weights.mean()),
        "cluster_weight_max": float(observed_weights.max()),
        "cluster_weight_variance": float(np.var(observed_weights)),
        "cluster_ess": effective_sample_size(observed_weights),
        "cluster_ess_fraction": effective_sample_size(observed_weights)
        / max(observed_weights.size, 1),
        "population_cluster_weight_max": float(weights.max()),
        "population_cluster_weight_variance": float(np.var(weights)),
        "unsupported_target_cluster_mass": float(pi_e_c[unsupported].sum() / pi_e_c.shape[0]),
        "clusters_without_any_behavior_support_fraction": float(
            np.mean(globally_unsupported)
        ),
        "clusters_with_contextual_zero_behavior_fraction": float(
            np.mean(contextually_zero)
        ),
        "complete_support_failure": bool(np.any(unsupported)),
        "mean_target_favored_cluster_mass": float(
            np.mean(pi_e_c[pi_e_c > pi_b_c + 1e-12])
        )
        if np.any(pi_e_c > pi_b_c + 1e-12)
        else 0.0,
    }


def effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def policy_value(q_population: np.ndarray, policy: np.ndarray) -> float:
    return float(np.sum(q_population * policy, axis=1).mean())


def fit_action_reward_model(bandit_data: Dict, random_state: int) -> np.ndarray:
    reg_model = RegressionModel(
        n_actions=bandit_data["n_actions"],
        action_context=bandit_data["action_context_one_hot"],
        base_model=MLP(
            hidden_layer_sizes=(50, 50, 50),
            random_state=random_state,
        ),
    )
    return reg_model.fit_predict(
        context=bandit_data["context"],
        action=bandit_data["action"],
        reward=bandit_data["reward"],
    )


def estimate_dr_dm(
    bandit_data: Dict,
    pi_e_logged: np.ndarray,
    q_x_a: np.ndarray,
) -> Dict[str, float]:
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=[DR(estimator_name="DR"), DM(estimator_name="DM")],
    )
    return ope.estimate_policy_values(
        action_dist=pi_e_logged,
        estimated_rewards_by_reg_model={"DR": q_x_a, "DM": q_x_a},
    )


def estimate_offcem(
    bandit_data: Dict,
    pi_e_logged: np.ndarray,
    clusters_3d: np.ndarray,
    f_x_a: np.ndarray,
) -> float:
    estimator_name = "OffCEM matched"
    observed_clusters = clusters_3d[bandit_data["user_idx"]]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=[
            OffCEM(
                n_actions=bandit_data["n_actions"],
                estimator_name=estimator_name,
                is_clustering=True,
            )
        ],
    )
    values = ope.estimate_policy_values(
        action_dist=pi_e_logged,
        action_embed=bandit_data["action_embed"],
        pi_b=bandit_data["pi_b"],
        p_e_a={estimator_name: observed_clusters},
        estimated_rewards_by_reg_model={estimator_name: f_x_a},
    )
    return float(values[estimator_name])


def run_seed(
    seed_index: int,
    n_rounds: int,
    n_users: int,
    n_actions: int,
    n_clusters: int,
    beta: float,
    reward_std: float,
    target_tau: float,
    target_strength: float,
    overlap_list: List[float],
) -> List[Dict]:
    seed = DEFAULTS["RANDOM_STATE"] + seed_index
    dataset = make_dataset(
        seed=seed,
        n_actions=n_actions,
        beta=beta,
        reward_std=reward_std,
    )
    base_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
        n_users=n_users,
        n_clusters=n_clusters,
        clustering_method="original",
        cluster_balance="natural",
        cluster_temperature=10.0,
    )
    q_population = base_data["fixed_expected_rewards"]
    clusters = base_data["cluster_indices"]
    clusters_3d = clusters_to_onehot_3d(clusters, base_data["n_users"])
    pi0_population = reconstruct_pi0_population(q_population, beta=beta)
    target_population = build_cluster_favoring_target(
        q_population=q_population,
        pi0_population=pi0_population,
        clusters=clusters,
        tau=target_tau,
        strength=target_strength,
    )
    true_value = policy_value(q_population, target_population)

    rows = []
    for overlap_level in overlap_list:
        behavior_population = degrade_behavior_cluster_overlap(
            pi0_population=pi0_population,
            target_population=target_population,
            clusters=clusters,
            overlap_level=overlap_level,
        )
        bandit_data = resample_logged_data(
            base_data=base_data,
            behavior_population=behavior_population,
            reward_std=reward_std,
            sample_seed=seed + int(round(overlap_level * 100000)) + 9001,
        )
        model_seed = seed + int(round(overlap_level * 100000)) + n_rounds
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        q_x_a = fit_action_reward_model(bandit_data, random_state=model_seed)
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        f_x_a = train_reward_model_via_two_stage(
            bandit_data=bandit_data,
            clusters=clusters_3d,
            need_q_x_a=False,
            random_state=model_seed,
        )
        pi_e_logged = target_population[bandit_data["user_idx"], :, np.newaxis]
        estimates = estimate_dr_dm(bandit_data, pi_e_logged, q_x_a)
        estimate_offcem_value = estimate_offcem(
            bandit_data=bandit_data,
            pi_e_logged=pi_e_logged,
            clusters_3d=clusters_3d,
            f_x_a=f_x_a,
        )
        overlap = cluster_overlap_diagnostics(
            behavior_population=behavior_population,
            target_population=target_population,
            clusters=clusters,
            user_idx=bandit_data["user_idx"],
            observed_actions=bandit_data["action"],
        )
        rows.append(
            result_row(
                seed_index=seed_index,
                overlap_level=overlap_level,
                target_tau=target_tau,
                target_strength=target_strength,
                estimate_offcem=estimate_offcem_value,
                estimate_dr=float(estimates["DR"]),
                estimate_dm=float(estimates["DM"]),
                true_value=true_value,
                overlap=overlap,
            )
        )
    return rows


def result_row(
    seed_index: int,
    overlap_level: float,
    target_tau: float,
    target_strength: float,
    estimate_offcem: float,
    estimate_dr: float,
    estimate_dm: float,
    true_value: float,
    overlap: Dict[str, float],
) -> Dict:
    error_offcem = float(estimate_offcem - true_value)
    error_dr = float(estimate_dr - true_value)
    error_dm = float(estimate_dm - true_value)
    return {
        "seed": int(seed_index),
        "partition": "matched",
        "overlap_level": float(overlap_level),
        "target_tau": float(target_tau),
        "target_strength": float(target_strength),
        "estimate_offcem": float(estimate_offcem),
        "estimate_dr": float(estimate_dr),
        "estimate_dm": float(estimate_dm),
        "true_value": float(true_value),
        "error_offcem": error_offcem,
        "error_dr": error_dr,
        "error_dm": error_dm,
        "sq_error_offcem": float(error_offcem**2),
        "sq_error_dr": float(error_dr**2),
        "sq_error_dm": float(error_dm**2),
        **overlap,
    }


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["overlap_level"], []).append(row)

    aggregates = []
    for overlap_level, items in sorted(grouped.items(), reverse=True):
        true_values = np.array([item["true_value"] for item in items])
        row = {
            "partition": "matched",
            "overlap_level": float(overlap_level),
            "target_tau": float(items[0]["target_tau"]),
            "target_strength": float(items[0]["target_strength"]),
            "true_value_mean": float(true_values.mean()),
            "n_seeds": len(items),
        }
        for key in (
            "cluster_weight_mean",
            "cluster_weight_max",
            "cluster_weight_variance",
            "cluster_ess",
            "cluster_ess_fraction",
            "population_cluster_weight_max",
            "population_cluster_weight_variance",
            "unsupported_target_cluster_mass",
            "clusters_without_any_behavior_support_fraction",
            "clusters_with_contextual_zero_behavior_fraction",
            "mean_target_favored_cluster_mass",
        ):
            row[f"{key}_mean"] = float(np.mean([item[key] for item in items]))
        row["complete_support_failure_rate"] = float(
            np.mean([item["complete_support_failure"] for item in items])
        )
        for estimator in ("offcem", "dr", "dm"):
            errors = np.array([item[f"error_{estimator}"] for item in items])
            estimates = np.array([item[f"estimate_{estimator}"] for item in items])
            row[f"estimate_{estimator}_mean"] = float(estimates.mean())
            row[f"mse_{estimator}"] = float(np.mean(errors**2))
            row[f"rel_mse_{estimator}"] = float(
                np.mean((errors**2) / np.maximum(true_values**2, 1e-12))
            )
            row[f"bias2_{estimator}"] = float(np.mean(errors) ** 2)
            row[f"variance_{estimator}"] = float(
                np.mean((errors - errors.mean()) ** 2)
            )
        row["offcem_mse_minus_dr"] = float(row["mse_offcem"] - row["mse_dr"])
        row["offcem_mse_minus_dm"] = float(row["mse_offcem"] - row["mse_dm"])
        aggregates.append(row)
    return aggregates


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_tidy_csv(path: Path) -> List[Dict]:
    with open(path) as file:
        return [
            {
                key: _parse_csv_value(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


def _parse_csv_value(value: str):
    if value in {"True", "False"}:
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def plot_aggregates(aggregates: List[Dict], out_dir: Path) -> None:
    x = [row["overlap_level"] for row in aggregates]
    metrics = [
        ("rel_mse", "Relative MSE", "rel_mse_vs_overlap.png"),
        ("variance", "Variance", "variance_vs_overlap.png"),
    ]
    for prefix, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for estimator, color in (
            ("offcem", "#1f77b4"),
            ("dr", "#d62728"),
            ("dm", "#9467bd"),
        ):
            ax.plot(
                x,
                [row[f"{prefix}_{estimator}"] for row in aggregates],
                marker="o",
                linewidth=1.4,
                markersize=4,
                label=estimator.upper() if estimator != "offcem" else "OffCEM",
                color=color,
            )
        ax.set_xscale("symlog", linthresh=0.01)
        ax.invert_xaxis()
        ax.set_xlabel("behavior mass multiplier on target-favored clusters")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [row["population_cluster_weight_max_mean"] for row in aggregates],
        marker="o",
        label="max population cluster weight",
    )
    ax.plot(
        x,
        [row["cluster_ess_fraction_mean"] for row in aggregates],
        marker="o",
        label="observed cluster ESS fraction",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.invert_xaxis()
    ax.set_xlabel("behavior mass multiplier on target-favored clusters")
    ax.set_ylabel("Overlap diagnostic")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_overlap_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args) -> None:
    if args.quick:
        args.n_rounds = 1000
        args.n_seeds = 2
        args.overlap_list = "1,.1"
        print("[quick] n_rounds=1000 n_seeds=2 overlap=1,.1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "cluster_overlap_sweep_tidy.csv"
    aggregate_path = out_dir / "cluster_overlap_sweep_aggregate.csv"

    if args.plot_only:
        rows = load_tidy_csv(tidy_path)
        aggregates = aggregate_results(rows)
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir)
        return

    overlap_list = parse_overlap_list(args.overlap_list)
    all_rows = []
    started = time()
    for seed_index in tqdm(range(args.n_seeds), desc="cluster-overlap seeds"):
        all_rows.extend(
            run_seed(
                seed_index=seed_index,
                n_rounds=args.n_rounds,
                n_users=args.n_users,
                n_actions=args.n_actions,
                n_clusters=args.n_clusters,
                beta=args.beta,
                reward_std=args.reward_std,
                target_tau=args.target_tau,
                target_strength=args.target_strength,
                overlap_list=overlap_list,
            )
        )
        with open(out_dir / "latest_rows.json", "w") as file:
            json.dump(all_rows, file, indent=2)

    aggregates = aggregate_results(all_rows)
    write_csv(tidy_path, all_rows)
    write_csv(aggregate_path, aggregates)
    if not args.no_plot:
        plot_aggregates(aggregates, out_dir)
    elapsed = (time() - started) / 60
    print(f"wrote {tidy_path}")
    print(f"wrote {aggregate_path}")
    if not args.no_plot:
        print(f"wrote plots to {out_dir}")
    print(f"done in {elapsed:.1f} min")


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
