"""Policy-disagreement sweep for OffCEM.

The experiment fixes one synthetic world, one logged dataset, one reward model
per partition, and then varies only the target policy inside clusters. Target
policies are constructed separately for each evaluated partition so cluster
masses, and therefore OffCEM cluster weights, remain fixed at one.

Run from ``src/synthetic``:

    conda run -n offcem python -m run_policy_disagreement_sweep --quick
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
from typing import Optional

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
from sklearn.metrics import adjusted_rand_score
from sklearn.neural_network import MLPRegressor as MLP
import torch
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from dataset import SyntheticBanditDataset
from ope import OffCEM
from ope import OffPolicyEvaluation
from ope import train_reward_model_via_two_stage
from run_sample_size_stress_test import build_feedback_prefix


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
    ESTIMATION_SEED_OFFSET=100000,
)

PARTITION_LABELS = {
    "matched": "matched",
    "kmeans": "kmeans",
}

ESTIMATOR_LABELS = {
    "offcem": "OffCEM",
    "dr": "DR",
    "dm": "DM",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep target-policy disagreement while holding training fixed"
    )
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-actions", type=int, default=DEFAULTS["N_ACTIONS"])
    parser.add_argument("--n-users", type=int, default=DEFAULTS["N_USERS"])
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=DEFAULTS["BETA"])
    parser.add_argument("--lambda-list", type=str, default="0,.25,.5,.75,1")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/policy_disagreement_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use n_rounds=1000, n_seeds=2, lambda=0,1",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate CSV and plots from tidy results",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_lambda_list(value: str) -> List[float]:
    lambdas = [float(item) for item in value.split(",") if item]
    if not lambdas:
        raise ValueError("--lambda-list must contain at least one value")
    if any(lam < 0.0 or lam > 1.0 for lam in lambdas):
        raise ValueError("lambda values must lie in [0, 1]")
    return sorted(set(lambdas))


def make_dataset(seed: int, n_actions: int, beta: float, reward_std: float):
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
    """Reconstruct the default continuous logger used by the synthetic dataset."""
    pi0 = softmax(beta * fixed_expected_rewards)
    return pi0[:, :, np.newaxis]


def cluster_mass(policy_3d: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    policy = policy_3d[:, :, 0]
    n_clusters = int(cluster_labels.max()) + 1
    masses = np.zeros((policy.shape[0], n_clusters), dtype=float)
    for c in range(n_clusters):
        masses[:, c] = policy[:, cluster_labels == c].sum(axis=1)
    return masses


def build_policy_disagreement_target(
    q_population: np.ndarray,
    pi0_population: np.ndarray,
    cluster_labels: np.ndarray,
    lambda_: float,
    tau: float,
) -> np.ndarray:
    """Build pi_lambda for one partition while preserving its cluster masses."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if lambda_ < 0.0 or lambda_ > 1.0:
        raise ValueError("lambda_ must lie in [0, 1]")

    pi0 = pi0_population[:, :, 0]
    pi_lambda = np.zeros_like(pi0, dtype=float)
    n_clusters = int(cluster_labels.max()) + 1

    for c in range(n_clusters):
        action_mask = cluster_labels == c
        if not np.any(action_mask):
            continue
        pi0_c = pi0[:, action_mask].sum(axis=1)
        if np.any(pi0_c <= 0):
            raise ValueError("pi0 must assign positive mass to every cluster")
        pi0_cond = pi0[:, action_mask] / pi0_c[:, np.newaxis]
        rho_cond = softmax(q_population[:, action_mask] / tau)
        mixed_cond = (1.0 - lambda_) * pi0_cond + lambda_ * rho_cond
        pi_lambda[:, action_mask] = pi0_c[:, np.newaxis] * mixed_cond

    pi_lambda_3d = pi_lambda[:, :, np.newaxis]
    assert_policy_valid(pi_lambda_3d)
    if not np.allclose(
        cluster_mass(pi_lambda_3d, cluster_labels),
        cluster_mass(pi0_population, cluster_labels),
        atol=1e-10,
    ):
        raise AssertionError("target policy must preserve partition cluster mass")
    return pi_lambda_3d


def assert_policy_valid(policy_3d: np.ndarray) -> None:
    if policy_3d.ndim != 3 or policy_3d.shape[2] != 1:
        raise AssertionError("policy must have shape (n_users, n_actions, 1)")
    if np.any(policy_3d < -1e-12):
        raise AssertionError("policy contains negative probabilities")
    if not np.allclose(policy_3d[:, :, 0].sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("policy rows must sum to one")


def policy_value(q_population: np.ndarray, policy_3d: np.ndarray) -> float:
    assert_policy_valid(policy_3d)
    return float(np.sum(q_population * policy_3d[:, :, 0], axis=1).mean())


def compute_policy_disagreement(
    pi0_population: np.ndarray,
    pi_lambda: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """Weighted mean TV distance between pi_lambda(.|x,c) and pi0(.|x,c)."""
    pi0 = pi0_population[:, :, 0]
    pi_e = pi_lambda[:, :, 0]
    n_clusters = int(cluster_labels.max()) + 1
    disagreement_by_user = np.zeros(pi0.shape[0], dtype=float)

    for c in range(n_clusters):
        action_mask = cluster_labels == c
        if not np.any(action_mask):
            continue
        pi0_c = pi0[:, action_mask].sum(axis=1)
        pi_e_c = pi_e[:, action_mask].sum(axis=1)
        pi0_cond = pi0[:, action_mask] / pi0_c[:, np.newaxis]
        pi_e_cond = pi_e[:, action_mask] / pi_e_c[:, np.newaxis]
        tv = 0.5 * np.abs(pi_e_cond - pi0_cond).sum(axis=1)
        disagreement_by_user += pi0_c * tv

    return float(disagreement_by_user.mean())


def compute_local_correctness_diagnostics(
    q_population: np.ndarray,
    f_hat: np.ndarray,
    pi0_population: np.ndarray,
    pi_e_population: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, float]:
    """Measure local error and target-specific bias pressure.

    Inputs must be aligned to the same context rows as ``f_hat``. ``dm_mse``
    uses uniform within-cluster centering and is policy-independent.
    ``lc_dm_mse_pi0`` centers errors by pi0(.|x,c). ``population_bias_formula``
    is sum_a (pi0 - pi_e) error, averaged over those context rows.
    """
    q = np.asarray(q_population, dtype=float)
    pred = np.asarray(f_hat, dtype=float)
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    pi0 = np.asarray(pi0_population, dtype=float)[:, :, 0]
    pi_e = np.asarray(pi_e_population, dtype=float)[:, :, 0]
    if q.shape != pred.shape or q.shape != pi0.shape or q.shape != pi_e.shape:
        raise ValueError("q, f_hat, pi0, and pi_e must have equal 2D shapes")
    if q.shape[1] != cluster_labels.size:
        raise ValueError("cluster_labels must contain one label per action")

    error = pred - q
    uniform_demeaned = np.zeros_like(error)
    logging_demeaned = np.zeros_like(error)
    pairwise_sum = 0.0
    pairwise_count = 0
    ratio_pairwise_sum = 0.0
    ratio_pairwise_count = 0
    n_clusters = int(cluster_labels.max()) + 1

    for c in range(n_clusters):
        action_mask = cluster_labels == c
        if not np.any(action_mask):
            continue
        cluster_error = error[:, action_mask]
        uniform_center = cluster_error.mean(axis=1)
        pi0_c = pi0[:, action_mask].sum(axis=1)
        pi0_cond = pi0[:, action_mask] / pi0_c[:, np.newaxis]
        logging_center = np.sum(pi0_cond * cluster_error, axis=1)
        uniform_demeaned[:, action_mask] = (
            cluster_error - uniform_center[:, np.newaxis]
        )
        logging_demeaned[:, action_mask] = (
            cluster_error - logging_center[:, np.newaxis]
        )
        if cluster_error.shape[1] >= 2:
            diffs = (
                cluster_error[:, :, np.newaxis]
                - cluster_error[:, np.newaxis, :]
            )
            upper = np.triu_indices(cluster_error.shape[1], k=1)
            pairwise_sq = diffs[:, upper[0], upper[1]] ** 2
            pairwise_sum += float(pairwise_sq.sum())
            pairwise_count += int(pairwise_sq.size)

            ratio = pi_e_cond = pi_e[:, action_mask] / pi_e[
                :, action_mask
            ].sum(axis=1)[:, np.newaxis]
            ratio = ratio / np.maximum(pi0_cond, 1e-12)
            ratio_diffs = ratio[:, :, np.newaxis] - ratio[:, np.newaxis, :]
            weighted_pairwise_sq = (
                ratio_diffs[:, upper[0], upper[1]] ** 2
                * pairwise_sq
            )
            ratio_pairwise_sum += float(weighted_pairwise_sq.sum())
            ratio_pairwise_count += int(weighted_pairwise_sq.size)

    bias_by_user = np.sum((pi0 - pi_e) * error, axis=1)
    population_bias_formula = float(bias_by_user.mean())
    return {
        "reward_mse": float(np.mean(error**2)),
        "lc_dm_mse_uniform": float(np.mean(uniform_demeaned**2)),
        "lc_dm_mse_pi0": float(np.mean(logging_demeaned**2)),
        "pairwise_lc_mse": (
            float(pairwise_sum / pairwise_count)
            if pairwise_count > 0
            else np.nan
        ),
        "within_cluster_ratio_pairwise_mse": (
            float(ratio_pairwise_sum / ratio_pairwise_count)
            if ratio_pairwise_count > 0
            else np.nan
        ),
        "population_bias_formula": population_bias_formula,
        "theorem33_bias": float(-population_bias_formula),
        "policy_lc_weighted_covariance": population_bias_formula,
    }


def build_fixed_partitions(
    bandit_data: Dict,
    n_clusters: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    estimation_seed = seed + DEFAULTS["ESTIMATION_SEED_OFFSET"]
    kmeans = compute_clusters(
        action_features=bandit_data["action_context_one_hot"],
        n_clusters=n_clusters,
        method="kmeans",
        balance="natural",
        random_state=estimation_seed,
    )
    return {
        "matched": bandit_data["cluster_indices"].copy(),
        "kmeans": kmeans,
    }


def fit_action_reward_model(
    bandit_data: Dict,
    random_state: int,
) -> np.ndarray:
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
        ope_estimators=[
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
        ],
    )
    return ope.estimate_policy_values(
        action_dist=pi_e_logged,
        estimated_rewards_by_reg_model={
            "DR": q_x_a,
            "DM": q_x_a,
        },
    )


def estimate_offcem(
    bandit_data: Dict,
    pi_e_logged: np.ndarray,
    clusters_3d: np.ndarray,
    f_x_a: np.ndarray,
    estimator_name: str,
) -> float:
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


def assert_cluster_weights_one(
    pi0_population: np.ndarray,
    pi_e_population: np.ndarray,
    cluster_labels: np.ndarray,
) -> None:
    pi0_mass = cluster_mass(pi0_population, cluster_labels)
    pie_mass = cluster_mass(pi_e_population, cluster_labels)
    if not np.allclose(pi0_mass, pie_mass, atol=1e-10):
        raise AssertionError("cluster-level importance weights are not one")


def cluster_weight_max_abs_dev_from_one(
    pi0_population: np.ndarray,
    pi_e_population: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    pi0_mass = cluster_mass(pi0_population, cluster_labels)
    pie_mass = cluster_mass(pi_e_population, cluster_labels)
    weights = pie_mass / np.maximum(pi0_mass, 1e-12)
    return float(np.max(np.abs(weights - 1.0)))


def run_seed(
    seed_index: int,
    n_rounds: int,
    n_users: int,
    n_actions: int,
    n_clusters: int,
    beta: float,
    reward_std: float,
    lambda_list: List[float],
    tau: float,
) -> List[Dict]:
    seed = DEFAULTS["RANDOM_STATE"] + seed_index
    dataset = make_dataset(
        seed=seed,
        n_actions=n_actions,
        beta=beta,
        reward_std=reward_std,
    )
    full_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
        n_users=n_users,
        n_clusters=n_clusters,
        clustering_method="original",
        cluster_balance="natural",
        cluster_temperature=10.0,
    )
    bandit_data = build_feedback_prefix(full_data, n_rounds)
    q_population = bandit_data["fixed_expected_rewards"]
    pi0_population = reconstruct_pi0_population(q_population, beta=beta)
    _assert_logger_matches_logged_rows(bandit_data, pi0_population)

    partitions = build_fixed_partitions(
        bandit_data=bandit_data,
        n_clusters=n_clusters,
        seed=seed,
    )
    ari_to_generating = {
        partition: float(
            adjusted_rand_score(
                bandit_data["cluster_indices"],
                labels,
            )
        )
        for partition, labels in partitions.items()
    }
    cluster_tensors = {
        name: clusters_to_onehot_3d(labels, bandit_data["n_users"])
        for name, labels in partitions.items()
    }

    model_seed = seed + n_rounds
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    q_x_a = fit_action_reward_model(bandit_data, random_state=model_seed)

    f_x_a_by_partition = {}
    for partition, clusters_3d in cluster_tensors.items():
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        f_x_a_by_partition[partition] = train_reward_model_via_two_stage(
            bandit_data=bandit_data,
            clusters=clusters_3d,
            need_q_x_a=False,
            random_state=model_seed,
        )

    rows = []
    for partition, labels in partitions.items():
        clusters_3d = cluster_tensors[partition]
        for lambda_ in lambda_list:
            pi_lambda = build_policy_disagreement_target(
                q_population=q_population,
                pi0_population=pi0_population,
                cluster_labels=labels,
                lambda_=lambda_,
                tau=tau,
            )
            assert_cluster_weights_one(pi0_population, pi_lambda, labels)
            true_value = policy_value(q_population, pi_lambda)
            disagreement = compute_policy_disagreement(
                pi0_population=pi0_population,
                pi_lambda=pi_lambda,
                cluster_labels=labels,
            )
            diagnostics = compute_local_correctness_diagnostics(
                q_population=bandit_data["expected_reward"],
                f_hat=f_x_a_by_partition[partition],
                pi0_population=pi0_population[bandit_data["user_idx"]],
                pi_e_population=pi_lambda[bandit_data["user_idx"]],
                cluster_labels=labels,
            )
            pi_e_logged = pi_lambda[bandit_data["user_idx"]]

            base_estimates = estimate_dr_dm(bandit_data, pi_e_logged, q_x_a)
            offcem_name = f"OffCEM {PARTITION_LABELS[partition]}"
            offcem_estimate = estimate_offcem(
                bandit_data=bandit_data,
                pi_e_logged=pi_e_logged,
                clusters_3d=clusters_3d,
                f_x_a=f_x_a_by_partition[partition],
                estimator_name=offcem_name,
            )
            rows.append(
                _result_row(
                    seed_index=seed_index,
                    partition=partition,
                    lambda_=lambda_,
                    tau=tau,
                    estimate_offcem=offcem_estimate,
                    estimate_dr=float(base_estimates["DR"]),
                    estimate_dm=float(base_estimates["DM"]),
                    true_value=true_value,
                    within_cluster_tv=disagreement,
                    cluster_weight_dev=cluster_weight_max_abs_dev_from_one(
                        pi0_population,
                        pi_lambda,
                        labels,
                    ),
                    ari_to_generating_partition=ari_to_generating[partition],
                    diagnostics=diagnostics,
                )
            )

    return rows


def _assert_logger_matches_logged_rows(
    bandit_data: Dict,
    pi0_population: np.ndarray,
) -> None:
    pi0_logged = pi0_population[bandit_data["user_idx"]]
    if not np.allclose(pi0_logged, bandit_data["pi_b"], atol=1e-10):
        raise AssertionError("reconstructed pi0 does not match logged pi_b")


def _result_row(
    seed_index: int,
    partition: str,
    lambda_: float,
    tau: float,
    estimate_offcem: float,
    estimate_dr: float,
    estimate_dm: float,
    true_value: float,
    within_cluster_tv: float,
    cluster_weight_dev: float,
    ari_to_generating_partition: float,
    diagnostics: Dict[str, float],
) -> Dict:
    error_offcem = float(estimate_offcem - true_value)
    error_dr = float(estimate_dr - true_value)
    error_dm = float(estimate_dm - true_value)
    row = {
        "seed": int(seed_index),
        "partition": partition,
        "lambda": float(lambda_),
        "tau": float(tau),
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
        "within_cluster_tv": float(within_cluster_tv),
        "within_cluster_ratio_pairwise_mse": diagnostics[
            "within_cluster_ratio_pairwise_mse"
        ],
        "cluster_weight_max_abs_dev_from_1": float(cluster_weight_dev),
        "reward_mse": diagnostics["reward_mse"],
        "lc_dm_mse_uniform": diagnostics["lc_dm_mse_uniform"],
        "lc_dm_mse_pi0": diagnostics["lc_dm_mse_pi0"],
        "pairwise_lc_mse": diagnostics["pairwise_lc_mse"],
        "population_bias_formula": diagnostics["population_bias_formula"],
        "theorem33_bias": diagnostics["theorem33_bias"],
        "policy_lc_weighted_covariance": diagnostics[
            "policy_lc_weighted_covariance"
        ],
        "ARI_to_generating_partition": float(ari_to_generating_partition),
    }
    return row


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["partition"], row["lambda"], row["tau"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (partition, lambda_, tau), items in sorted(grouped.items()):
        true_values = np.array([item["true_value"] for item in items], dtype=float)
        row = {
            "partition": partition,
            "lambda": float(lambda_),
            "tau": float(tau),
            "true_value_mean": float(true_values.mean()),
            "within_cluster_tv_mean": _mean(items, "within_cluster_tv"),
            "within_cluster_ratio_pairwise_mse_mean": _mean(
                items,
                "within_cluster_ratio_pairwise_mse",
            ),
            "cluster_weight_max_abs_dev_from_1_max": float(
                np.max(
                    [
                        item["cluster_weight_max_abs_dev_from_1"]
                        for item in items
                    ]
                )
            ),
            "reward_mse_mean": _mean(items, "reward_mse"),
            "lc_dm_mse_uniform_mean": _mean(items, "lc_dm_mse_uniform"),
            "lc_dm_mse_pi0_mean": _mean(items, "lc_dm_mse_pi0"),
            "pairwise_lc_mse_mean": _mean(items, "pairwise_lc_mse"),
            "population_bias_formula_mean": _mean(
                items,
                "population_bias_formula",
            ),
            "theorem33_bias_mean": _mean(items, "theorem33_bias"),
            "policy_lc_weighted_covariance_mean": _mean(
                items,
                "policy_lc_weighted_covariance",
            ),
            "ARI_to_generating_partition_mean": _mean(
                items,
                "ARI_to_generating_partition",
            ),
            "n_seeds": len(items),
        }
        for estimator in ("offcem", "dr", "dm"):
            estimates = np.array(
                [item[f"estimate_{estimator}"] for item in items],
                dtype=float,
            )
            errors = np.array(
                [item[f"error_{estimator}"] for item in items],
                dtype=float,
            )
            row[f"estimate_{estimator}_mean"] = float(estimates.mean())
            row[f"mse_{estimator}"] = float(np.mean(errors**2))
            row[f"rel_mse_{estimator}"] = float(
                np.mean((errors**2) / np.maximum(true_values**2, 1e-12))
            )
            row[f"bias2_{estimator}"] = float(np.mean(errors) ** 2)
            row[f"variance_{estimator}"] = float(
                np.mean((errors - errors.mean()) ** 2)
            )
        aggregates.append(row)
    return aggregates


def _mean(items: List[Dict], key: str) -> float:
    values = np.array([item[key] for item in items], dtype=float)
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else np.nan


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
                "seed": int(row["seed"]),
                "partition": row["partition"],
                "lambda": float(row["lambda"]),
                "tau": float(row["tau"]),
                "estimate_offcem": float(row["estimate_offcem"]),
                "estimate_dr": float(row["estimate_dr"]),
                "estimate_dm": float(row["estimate_dm"]),
                "true_value": float(row["true_value"]),
                "error_offcem": float(row["error_offcem"]),
                "error_dr": float(row["error_dr"]),
                "error_dm": float(row["error_dm"]),
                "sq_error_offcem": float(row["sq_error_offcem"]),
                "sq_error_dr": float(row["sq_error_dr"]),
                "sq_error_dm": float(row["sq_error_dm"]),
                "within_cluster_tv": float(row["within_cluster_tv"]),
                "within_cluster_ratio_pairwise_mse": float(
                    row["within_cluster_ratio_pairwise_mse"]
                ),
                "cluster_weight_max_abs_dev_from_1": float(
                    row["cluster_weight_max_abs_dev_from_1"]
                ),
                "reward_mse": float(row["reward_mse"]),
                "lc_dm_mse_uniform": float(row["lc_dm_mse_uniform"]),
                "lc_dm_mse_pi0": float(row["lc_dm_mse_pi0"]),
                "pairwise_lc_mse": float(row["pairwise_lc_mse"]),
                "population_bias_formula": float(
                    row["population_bias_formula"]
                ),
                "theorem33_bias": float(row["theorem33_bias"]),
                "policy_lc_weighted_covariance": float(
                    row["policy_lc_weighted_covariance"]
                ),
                "ARI_to_generating_partition": float(
                    row["ARI_to_generating_partition"]
                ),
            }
            for row in csv.DictReader(file)
        ]


def plot_aggregates(aggregates: List[Dict], out_dir: Path) -> None:
    metrics = [
        ("bias2", "Bias^2", "bias2_vs_lambda.png"),
        ("rel_mse", "Relative MSE", "rel_mse_vs_lambda.png"),
    ]
    colors = {
        ("matched", "offcem"): "#1f77b4",
        ("matched", "dr"): "#d62728",
        ("matched", "dm"): "#9467bd",
        ("kmeans", "offcem"): "#2ca02c",
        ("kmeans", "dr"): "#ff7f0e",
        ("kmeans", "dm"): "#8c564b",
    }
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for partition in ("matched", "kmeans"):
            for estimator in ("offcem", "dr", "dm"):
                selected = [
                    row
                    for row in aggregates
                    if row["partition"] == partition
                ]
                if not selected:
                    continue
                selected = sorted(selected, key=lambda row: row["lambda"])
                label = f"{ESTIMATOR_LABELS[estimator]} {partition}"
                ax.plot(
                    [row["lambda"] for row in selected],
                    [row[f"{key}_{estimator}"] for row in selected],
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    label=label,
                    color=colors.get((partition, estimator)),
                )
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for partition in ("matched", "kmeans"):
        selected = [
            row
            for row in aggregates
            if row["partition"] == partition
        ]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: row["lambda"])
        ax.plot(
            [row["lambda"] for row in selected],
            [row["within_cluster_tv_mean"] for row in selected],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=partition,
            color=colors.get((partition, "offcem")),
        )
    ax.set_xlabel("lambda")
    ax.set_ylabel("Policy disagreement")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out_dir / "policy_disagreement_vs_lambda.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, linestyle in (
        ("lc_dm_mse_uniform_mean", "-"),
        ("lc_dm_mse_pi0_mean", "--"),
        ("theorem33_bias_mean", ":"),
    ):
        for partition in ("matched", "kmeans"):
            selected = [
                row
                for row in aggregates
                if row["partition"] == partition
            ]
            if not selected:
                continue
            selected = sorted(selected, key=lambda row: row["lambda"])
            ax.plot(
                [row["lambda"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                linewidth=1.4,
                markersize=4,
                linestyle=linestyle,
                label=f"{partition} {metric.replace('_mean', '')}",
            )
    ax.set_xlabel("lambda")
    ax.set_ylabel("Local-correctness diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        out_dir / "local_correctness_diagnostics_vs_lambda.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_experiment(args) -> None:
    if args.quick:
        args.n_rounds = 1000
        args.n_seeds = 2
        args.lambda_list = "0,1"
        print("[quick] n_rounds=1000 n_seeds=2 lambda=0,1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "policy_disagreement_sweep_tidy.csv"
    aggregate_path = out_dir / "policy_disagreement_sweep_aggregate.csv"

    if args.plot_only:
        rows = load_tidy_csv(tidy_path)
        aggregates = aggregate_results(rows)
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir)
        return

    lambda_list = parse_lambda_list(args.lambda_list)
    all_rows = []
    started = time()
    for seed_index in tqdm(range(args.n_seeds), desc="policy-disagreement seeds"):
        all_rows.extend(
            run_seed(
                seed_index=seed_index,
                n_rounds=args.n_rounds,
                n_users=args.n_users,
                n_actions=args.n_actions,
                n_clusters=args.n_clusters,
                beta=args.beta,
                reward_std=args.reward_std,
                lambda_list=lambda_list,
                tau=args.tau,
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
