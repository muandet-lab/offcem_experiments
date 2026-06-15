"""Diagnostics for explaining OffCEM win/loss regions."""
from typing import Dict

import numpy as np
from sklearn.metrics import adjusted_rand_score


def demeaned_error(
    prediction: np.ndarray,
    expected_reward: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    error = prediction - expected_reward
    centered = error.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[:, mask] -= error[:, mask].mean(axis=1, keepdims=True)
    return centered


def reward_diagnostics(
    prediction_2s: np.ndarray,
    prediction_1s: np.ndarray,
    expected_reward: np.ndarray,
    clusters: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    reward_variance: float,
) -> Dict[str, float]:
    dm2_error = demeaned_error(prediction_2s, expected_reward, clusters)
    dm1_error = demeaned_error(prediction_1s, expected_reward, clusters)
    dm2 = float(np.mean(dm2_error**2))
    dm1 = float(np.mean(dm1_error**2))

    cluster_matrix = np.zeros((clusters.size, int(clusters.max()) + 1))
    cluster_matrix[np.arange(clusters.size), clusters] = 1.0
    pi_b_c = pi_b @ cluster_matrix
    pi_e_c = pi_e @ cluster_matrix
    observed_cluster_weight = pi_e_c / np.maximum(pi_b_c, 1e-12)
    coefficient = np.empty_like(pi_e)
    for action, cluster in enumerate(clusters):
        coefficient[:, action] = (
            pi_b[:, action] * observed_cluster_weight[:, cluster]
            - pi_e[:, action]
        )
    weighted_dm = float(np.mean((coefficient * dm2_error) ** 2))

    return {
        "dm_2s": dm2,
        "dm_1s": dm1,
        "dm_ratio": dm2 / max(dm1, 1e-12),
        "dm_variance_scaled": dm2 / max(reward_variance, 1e-12),
        "dm_policy_weighted": weighted_dm,
        "reward_mse_2s": float(np.mean((prediction_2s - expected_reward) ** 2)),
        "reward_mse_1s": float(np.mean((prediction_1s - expected_reward) ** 2)),
    }


def overlap_diagnostics(
    bandit_data: Dict,
    pi_e_rows: np.ndarray,
    clusters: np.ndarray,
) -> Dict[str, float]:
    actions = bandit_data["action"]
    pscore = bandit_data["pscore"]
    action_weights = pi_e_rows[np.arange(actions.size), actions] / np.maximum(
        pscore, 1e-12
    )

    population_pi_b = bandit_data["pi_b_population"]
    population_pi_e = bandit_data["target_policy_population"]
    n_clusters = int(clusters.max()) + 1
    matrix = np.zeros((clusters.size, n_clusters))
    matrix[np.arange(clusters.size), clusters] = 1.0
    pi_b_c = population_pi_b @ matrix
    pi_e_c = population_pi_e @ matrix
    observed_clusters = clusters[actions]
    cluster_weights = (
        pi_e_c[bandit_data["user_idx"], observed_clusters]
        / np.maximum(
            pi_b_c[bandit_data["user_idx"], observed_clusters], 1e-12
        )
    )

    support = bandit_data["support_mask"]
    unsupported_target_mass = float(population_pi_e[:, ~support].sum(1).mean())
    cluster_sizes = np.bincount(clusters, minlength=n_clusters)
    supported_sizes = np.bincount(
        clusters[support], minlength=n_clusters
    )
    observed_action = np.bincount(actions, minlength=clusters.size) > 0
    observed_per_cluster = np.bincount(
        clusters[observed_action], minlength=n_clusters
    )

    pair_count = 0
    for user in range(bandit_data["n_users"]):
        observed = np.flatnonzero(bandit_data["obs_mat"][user])
        if observed.size == 0:
            continue
        counts = np.bincount(clusters[observed], minlength=n_clusters)
        pair_count += int(np.sum(counts * np.maximum(counts - 1, 0)))

    return {
        "action_ess": _effective_sample_size(action_weights),
        "cluster_ess": _effective_sample_size(cluster_weights),
        "action_weight_max": float(np.max(action_weights)),
        "action_weight_variance": float(np.var(action_weights)),
        "cluster_weight_max": float(np.max(cluster_weights)),
        "cluster_weight_variance": float(np.var(cluster_weights)),
        "unsupported_target_mass": unsupported_target_mass,
        "clusters_without_support_fraction": float(np.mean(supported_sizes == 0)),
        "supported_actions_per_cluster_min": int(supported_sizes.min()),
        "observed_actions_per_cluster_mean": float(observed_per_cluster.mean()),
        "observed_actions_per_cluster_min": int(observed_per_cluster.min()),
        "pairwise_comparisons": pair_count,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_cv": float(
            cluster_sizes.std() / max(cluster_sizes.mean(), 1e-12)
        ),
    }


def partition_diagnostics(
    generating_clusters: np.ndarray,
    estimation_clusters: np.ndarray,
) -> Dict[str, float]:
    return {
        "ari_generating_estimation": float(
            adjusted_rand_score(generating_clusters, estimation_clusters)
        )
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)
