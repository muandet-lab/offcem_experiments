"""Focused sample-size stress test for OffCEM.

The experiment fixes one synthetic world per seed, generates a single logged
stream of length max(n_list), and evaluates nested prefixes. Only logged sample
size varies within a seed.

Run from ``src/synthetic``:

    conda run -n offcem python -m run_sample_size_stress_test --quick
"""
import argparse
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
from sklearn.neural_network import MLPRegressor as MLP
import torch
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from clustering import corrupt_partition
from dataset import SyntheticBanditDataset
from ope import OffCEM
from ope import OffPolicyEvaluation
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy
from sklearn.metrics import adjusted_rand_score


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
    STREAM_SEED_OFFSET=200000,
    NUISANCE_STREAM_SEED_OFFSET=300000,
    MODEL_SEED_OFFSET=400000,
)

ROUND_LEVEL_FIELDS = (
    "user_idx",
    "context",
    "action",
    "reward",
    "expected_reward",
    "pi_b",
    "pscore",
    "action_embed",
)

WORLD_LEVEL_FIELDS = (
    "world_seed",
    "stream_seed",
    "n_users",
    "n_actions",
    "clusters",
    "cluster_indices",
    "action_context",
    "action_context_one_hot",
    "fixed_user_contexts",
    "fixed_expected_rewards",
    "g_x_e",
    "p_e_a",
    "position",
)

ESTIMATOR_LABELS = {
    "matched": "OffCEM matched",
    "reestimated_wfss": "OffCEM reestimated WFSS",
    "kmeans": "OffCEM K-means",
    "DR": "DR",
    "DM": "DM",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample-size stress test with one fixed synthetic world per seed"
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="500,1000,3000,10000,30000,100000",
        help="Comma-separated logged prefix sizes",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--n-worlds",
        type=int,
        default=None,
        help="Number of independent worlds; defaults to --n-seeds for compatibility",
    )
    parser.add_argument(
        "--n-streams",
        type=int,
        default=1,
        help="Independent logged streams sampled per fixed world",
    )
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument(
        "--partitions",
        type=str,
        default="matched,wfss,kmeans,corrupt",
        help="Comma-separated: matched,wfss,kmeans,corrupt",
    )
    parser.add_argument(
        "--corruption-levels",
        type=str,
        default="0,0.1,0.25,0.5,0.75,1",
        help="Comma-separated matched-partition corruption probabilities",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="end_to_end",
        help="Comma-separated: end_to_end,frozen",
    )
    parser.add_argument(
        "--nuisance-train-rounds",
        type=int,
        default=30000,
        help="Independent training-stream size for the frozen-nuisance arm",
    )
    parser.add_argument(
        "--base-model-seed",
        type=int,
        default=DEFAULTS["MODEL_SEED_OFFSET"],
        help="Model RNG offset; intentionally independent of n and stream",
    )
    parser.add_argument(
        "--compact-evaluation",
        action="store_true",
        help=(
            "Avoid dense n_rounds x n_actions policy/reward tensors and use "
            "the equivalent population-indexed DR/DM/OffCEM score formulas. "
            "Required for practical million-round runs."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/sample_size_stress_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use n=500,3000 and 2 seeds",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate CSV and plots from tidy results",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_n_list(value: str) -> List[int]:
    n_list = [int(item.replace("_", "")) for item in value.split(",") if item]
    if not n_list:
        raise ValueError("--n-list must contain at least one value")
    if any(n <= 0 for n in n_list):
        raise ValueError("all n values must be positive")
    return sorted(set(n_list))


def parse_name_list(value: str, name: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"--{name} must contain at least one value")
    return items


def parse_float_list(value: str, name: str) -> List[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one value")
    return values


def make_dataset(seed: int, reward_std: float) -> SyntheticBanditDataset:
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
        reward_std=reward_std,
        random_state=seed,
    )


def population_policy_value(
    fixed_expected_rewards: np.ndarray,
    eps: float,
) -> tuple[float, np.ndarray]:
    pi_e_population = gen_eps_greedy(
        expected_reward=fixed_expected_rewards,
        eps=eps,
    )
    true_value = float(
        np.average(
            fixed_expected_rewards,
            weights=pi_e_population[:, :, 0],
            axis=1,
        ).mean()
    )
    return true_value, pi_e_population


def generate_world(
    world_seed: int,
    n_clusters: int,
    eps: float,
    reward_std: float,
) -> Dict:
    """Generate one fixed population and attach its target-policy value."""
    dataset = make_dataset(seed=world_seed, reward_std=reward_std)
    world = dataset.generate_fixed_world(
        n_users=DEFAULTS["N_USERS"],
        n_clusters=n_clusters,
        clustering_method="original",
        cluster_balance="natural",
        cluster_temperature=10.0,
        world_seed=world_seed,
    )
    true_value, pi_e_population = population_policy_value(
        world["fixed_expected_rewards"], eps=eps
    )
    world["pi_e_population"] = pi_e_population
    world["true_value"] = true_value
    return world


def sample_logged_stream(
    world: Dict,
    n_rounds: int,
    stream_seed: int,
    reward_std: float,
    compact: bool = False,
) -> Dict:
    """Sample one independent logged stream without changing the world."""
    dataset = make_dataset(seed=int(world["world_seed"]), reward_std=reward_std)
    return dataset.sample_logged_stream(
        world=world,
        n_rounds=n_rounds,
        stream_seed=stream_seed,
        materialize_round_policy=not compact,
    )


def build_feedback_prefix(full_bandit_data: Dict, n: int) -> Dict:
    """Build a nested logged-data prefix while preserving the synthetic world.

    Round-level fields are sliced to the first ``n`` interactions. World-level
    fields are reused. User-action reward matrices are rebuilt from the prefix
    with mean aggregation for repeated observations.
    """
    if n <= 0 or n > int(full_bandit_data["n_rounds"]):
        raise ValueError("n must be in [1, full_bandit_data['n_rounds']]")

    prefix = {"n_rounds": int(n)}
    for key in WORLD_LEVEL_FIELDS:
        if key in full_bandit_data:
            prefix[key] = full_bandit_data[key]
    for key in ROUND_LEVEL_FIELDS:
        if key not in full_bandit_data:
            continue
        value = full_bandit_data[key]
        if value is None:
            prefix[key] = None
        else:
            prefix[key] = value[:n].copy()

    n_users = int(prefix["n_users"])
    n_actions = int(prefix["n_actions"])
    reward_sum_mat = np.zeros((n_users, n_actions), dtype=float)
    obs_count_mat = np.zeros((n_users, n_actions), dtype=int)
    for user, action, reward in zip(
        prefix["user_idx"],
        prefix["action"],
        prefix["reward"],
    ):
        reward_sum_mat[int(user), int(action)] += float(reward)
        obs_count_mat[int(user), int(action)] += 1

    reward_mat = np.zeros((n_users, n_actions), dtype=float)
    observed = obs_count_mat > 0
    reward_mat[observed] = reward_sum_mat[observed] / obs_count_mat[observed]

    prefix["reward_sum_mat"] = reward_sum_mat
    prefix["obs_count_mat"] = obs_count_mat
    prefix["reward_mat"] = reward_mat
    prefix["obs_mat"] = observed.astype(int)

    _assert_prefix_shapes(prefix, n)
    if int(prefix["obs_count_mat"].sum()) != n:
        raise AssertionError("obs_count_mat.sum() must equal prefix length")
    return prefix


def _assert_prefix_shapes(prefix: Dict, n: int) -> None:
    for key in ROUND_LEVEL_FIELDS:
        if key not in prefix or prefix[key] is None:
            continue
        if prefix[key].shape[0] != n:
            raise AssertionError(
                f"{key}.shape[0] must be {n}, got {prefix[key].shape[0]}"
            )


def build_fixed_estimation_partitions(
    full_bandit_data: Dict,
    n_clusters: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    estimation_seed = seed + DEFAULTS["ESTIMATION_SEED_OFFSET"]
    reestimated = compute_clusters(
        action_features=full_bandit_data["action_context_one_hot"],
        n_clusters=n_clusters,
        method="original",
        balance="natural",
        random_state=estimation_seed,
        temperature=10.0,
    )
    kmeans = compute_clusters(
        action_features=full_bandit_data["action_context_one_hot"],
        n_clusters=n_clusters,
        method="kmeans",
        balance="natural",
        random_state=estimation_seed,
    )
    return {
        "matched": full_bandit_data["cluster_indices"].copy(),
        "reestimated_wfss": reestimated,
        "kmeans": kmeans,
    }


def build_partition_sweep(
    world: Dict,
    n_clusters: int,
    partition_seed: int,
    partition_names: List[str],
    corruption_levels: List[float],
) -> Dict[str, Dict]:
    """Build fixed estimation partitions and their quality metadata per world."""
    valid = {"matched", "wfss", "kmeans", "corrupt"}
    unknown = set(partition_names) - valid
    if unknown:
        raise ValueError(f"unknown partition names: {sorted(unknown)}")
    if any(level < 0.0 or level > 1.0 for level in corruption_levels):
        raise ValueError("--corruption-levels values must lie in [0, 1]")

    matched = world["cluster_indices"].copy()
    output = {}
    for name in partition_names:
        if name == "matched":
            output["matched"] = {"labels": matched, "corruption_level": 0.0}
        elif name == "wfss":
            output["wfss"] = {
                "labels": compute_clusters(
                    world["action_context_one_hot"],
                    n_clusters=n_clusters,
                    method="original",
                    balance="natural",
                    random_state=partition_seed,
                    temperature=10.0,
                ),
                "corruption_level": np.nan,
            }
        elif name == "kmeans":
            output["kmeans"] = {
                "labels": compute_clusters(
                    world["action_context_one_hot"],
                    n_clusters=n_clusters,
                    method="kmeans",
                    balance="natural",
                    random_state=partition_seed,
                ),
                "corruption_level": np.nan,
            }
        else:
            for level in corruption_levels:
                key = f"matched_corrupt_{level:g}"
                output[key] = {
                    "labels": corrupt_partition(
                        matched,
                        p=level,
                        random_state=partition_seed + int(round(level * 1_000_000)),
                    ),
                    "corruption_level": float(level),
                }
    for item in output.values():
        item["ari_to_matched"] = float(adjusted_rand_score(matched, item["labels"]))
    return output


def fit_action_reward_model(
    bandit_data: Dict,
    random_state: int,
    prediction_context: np.ndarray = None,
) -> np.ndarray:
    reg_model = RegressionModel(
        n_actions=bandit_data["n_actions"],
        action_context=bandit_data["action_context_one_hot"],
        base_model=MLP(
            hidden_layer_sizes=(50, 50, 50),
            random_state=random_state,
        ),
    )
    if prediction_context is None:
        return reg_model.fit_predict(
            context=bandit_data["context"],
            action=bandit_data["action"],
            reward=bandit_data["reward"],
        )
    reg_model.fit(
        context=bandit_data["context"],
        action=bandit_data["action"],
        reward=bandit_data["reward"],
    )
    return reg_model.predict(context=prediction_context)


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


def estimate_dr_dm_compact(
    bandit_data: Dict,
    world: Dict,
    q_population: np.ndarray,
) -> Dict[str, float]:
    """Compute the DR/DM means without materializing round-by-action tensors."""
    q = q_population[:, :, 0]
    pi_e = world["pi_e_population"][:, :, 0]
    user_idx = bandit_data["user_idx"]
    action = bandit_data["action"]
    dm_by_user = np.sum(pi_e * q, axis=1)
    dm_round = dm_by_user[user_idx]
    factual_q = q[user_idx, action]
    factual_pi_e = pi_e[user_idx, action]
    importance_weight = factual_pi_e / np.maximum(bandit_data["pscore"], 1e-12)
    dr_round = dm_round + importance_weight * (bandit_data["reward"] - factual_q)
    return {"DR": float(dr_round.mean()), "DM": float(dm_round.mean())}


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


def estimate_offcem_compact(
    bandit_data: Dict,
    world: Dict,
    cluster_labels: np.ndarray,
    f_population: np.ndarray,
) -> float:
    """Population-indexed implementation of the OffCEM sample score.

    This is algebraically the clustered branch of ``OffCEM._estimate_round_rewards``
    and avoids allocating ``(n_rounds, n_actions)`` arrays.
    """
    f = f_population[:, :, 0]
    pi_b = world["pi_b_population"][:, :, 0]
    pi_e = world["pi_e_population"][:, :, 0]
    n_users = int(world["n_users"])
    n_clusters = int(cluster_labels.max()) + 1
    pi_b_cluster = np.zeros((n_users, n_clusters))
    pi_e_cluster = np.zeros_like(pi_b_cluster)
    for cluster in range(n_clusters):
        mask = cluster_labels == cluster
        pi_b_cluster[:, cluster] = pi_b[:, mask].sum(axis=1)
        pi_e_cluster[:, cluster] = pi_e[:, mask].sum(axis=1)
    user_idx = bandit_data["user_idx"]
    action = bandit_data["action"]
    observed_cluster = cluster_labels[action]
    weights = (
        pi_e_cluster[user_idx, observed_cluster]
        / np.maximum(pi_b_cluster[user_idx, observed_cluster], 1e-12)
    )
    plugin_by_user = np.sum(pi_e * f, axis=1)
    score = plugin_by_user[user_idx]
    score += weights * (bandit_data["reward"] - f[user_idx, action])
    return float(score.mean())


def _set_model_seed(model_seed: int) -> None:
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)


def _local_correctness_diagnostics(
    world: Dict,
    f_population: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    q = world["fixed_expected_rewards"]
    f = f_population[:, :, 0]
    pi0 = world["pi_b_population"][:, :, 0]
    error = f - q
    demeaned = np.zeros_like(error)
    for cluster in np.unique(labels):
        mask = labels == cluster
        pi0_cond = pi0[:, mask] / pi0[:, mask].sum(axis=1, keepdims=True)
        center = (pi0_cond * error[:, mask]).sum(axis=1, keepdims=True)
        demeaned[:, mask] = error[:, mask] - center
    cluster_sizes = np.bincount(labels)
    return {
        "reward_mse": float(np.mean(error**2)),
        "lc_error": float(np.mean(demeaned**2)),
        "ari_to_matched": float(adjusted_rand_score(world["cluster_indices"], labels)),
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_std": float(cluster_sizes.std()),
    }


def _coverage_diagnostics(
    prefix: Dict,
    world: Dict,
    labels: np.ndarray,
) -> Dict[str, float]:
    n_users = int(world["n_users"])
    n_actions = int(world["n_actions"])
    pairs = np.unique(
        np.column_stack((prefix["user_idx"], prefix["action"])), axis=0
    )
    best_actions = world["fixed_expected_rewards"].argmax(axis=1)
    best_seen = prefix["obs_mat"][np.arange(n_users), best_actions] > 0
    target_mass_observed = np.sum(
        world["pi_e_population"][:, :, 0] * prefix["obs_mat"], axis=1
    ).mean()
    observed_cluster_cells = np.unique(
        np.column_stack((prefix["user_idx"], labels[prefix["action"]])), axis=0
    ).shape[0]
    action_counts = np.bincount(prefix["action"], minlength=n_actions)
    pairwise_examples = 0
    for user in range(n_users):
        observed_actions = np.flatnonzero(prefix["obs_mat"][user])
        if observed_actions.size == 0:
            continue
        counts = np.bincount(labels[observed_actions], minlength=int(labels.max()) + 1)
        pairwise_examples += int(np.sum(counts * (counts - 1)))

    pi0 = world["pi_b_population"][:, :, 0]
    pi_e = world["pi_e_population"][:, :, 0]
    cluster_weights = np.zeros((n_users, int(labels.max()) + 1))
    for cluster in range(cluster_weights.shape[1]):
        mask = labels == cluster
        cluster_weights[:, cluster] = (
            pi_e[:, mask].sum(axis=1) / np.maximum(pi0[:, mask].sum(axis=1), 1e-12)
        )
    logged_weights = cluster_weights[
        prefix["user_idx"], labels[prefix["action"]]
    ]
    weight_sum_sq = float(np.sum(logged_weights**2))
    cluster_ess = (
        float(np.sum(logged_weights) ** 2 / weight_sum_sq)
        if weight_sum_sq > 0
        else 0.0
    )
    return {
        "unique_users": int(np.unique(prefix["user_idx"]).size),
        "unique_actions": int(np.count_nonzero(action_counts)),
        "unique_user_action_pairs": int(pairs.shape[0]),
        "user_action_coverage": float(pairs.shape[0] / (n_users * n_actions)),
        "target_best_action_observation_rate": float(best_seen.mean()),
        "target_mass_on_observed_actions": float(target_mass_observed),
        "observed_user_cluster_cells": int(observed_cluster_cells),
        "pairwise_training_examples": int(pairwise_examples),
        "cluster_ess": cluster_ess,
        "cluster_ess_fraction": float(cluster_ess / prefix["n_rounds"]),
        "cluster_weight_max": float(logged_weights.max()),
        "cluster_weight_variance": float(logged_weights.var()),
    }


def _append_estimates(
    rows: List[Dict],
    arm: str,
    prefix: Dict,
    world: Dict,
    stream_index: int,
    model_seed: int,
    partition_sweep: Dict[str, Dict],
    q_population: np.ndarray,
    f_populations: Dict[str, np.ndarray],
    diagnostics: Dict[str, Dict],
    compact_evaluation: bool,
) -> None:
    n = int(prefix["n_rounds"])
    if compact_evaluation:
        base_estimates = estimate_dr_dm_compact(prefix, world, q_population)
    else:
        pi_e_logged = world["pi_e_population"][prefix["user_idx"]]
        q_logged = q_population[prefix["user_idx"]]
        base_estimates = estimate_dr_dm(prefix, pi_e_logged, q_logged)
    base_metadata = dict(
        world_seed=int(world["world_seed"]),
        stream_seed=int(prefix["stream_seed"]),
        stream_index=int(stream_index),
        model_seed=int(model_seed),
        arm=arm,
        partition="baseline",
        corruption_level=np.nan,
    )
    for estimator in ("DR", "DM"):
        rows.append(
            _result_row(
                n=n,
                seed_index=int(world["world_seed"]),
                estimator=estimator,
                estimate=float(base_estimates[estimator]),
                true_value=float(world["true_value"]),
                **base_metadata,
            )
        )

    for partition, item in partition_sweep.items():
        labels = item["labels"]
        if compact_evaluation:
            estimate = estimate_offcem_compact(
                prefix, world, labels, f_populations[partition]
            )
        else:
            clusters_3d = clusters_to_onehot_3d(labels, int(world["n_users"]))
            estimate = estimate_offcem(
                bandit_data=prefix,
                pi_e_logged=pi_e_logged,
                clusters_3d=clusters_3d,
                f_x_a=f_populations[partition][prefix["user_idx"]],
                estimator_name=f"OffCEM {partition}",
            )
        metadata = dict(base_metadata)
        metadata.update(
            partition=partition,
            corruption_level=item["corruption_level"],
            **diagnostics[partition],
            **_coverage_diagnostics(prefix, world, labels),
        )
        rows.append(
            _result_row(
                n=n,
                seed_index=int(world["world_seed"]),
                estimator=f"OffCEM {partition}",
                estimate=estimate,
                true_value=float(world["true_value"]),
                **metadata,
            )
        )


def run_world(
    world_index: int,
    n_list: List[int],
    n_clusters: int,
    eps: float,
    reward_std: float,
    n_streams: int,
    partition_names: List[str],
    corruption_levels: List[float],
    arms: List[str],
    nuisance_train_rounds: int,
    base_model_seed: int,
    compact_evaluation: bool,
    progress=None,
) -> List[Dict]:
    world_seed = DEFAULTS["RANDOM_STATE"] + world_index
    world = generate_world(world_seed, n_clusters, eps, reward_std)
    partition_sweep = build_partition_sweep(
        world,
        n_clusters=n_clusters,
        partition_seed=world_seed + DEFAULTS["ESTIMATION_SEED_OFFSET"],
        partition_names=partition_names,
        corruption_levels=corruption_levels,
    )
    model_seed = int(base_model_seed + world_index)
    rows = []

    frozen_models = None
    if "frozen" in arms:
        if progress is not None:
            progress.set_postfix_str(
                f"world={world_index + 1} training frozen nuisance models"
            )
        nuisance_stream = sample_logged_stream(
            world,
            n_rounds=nuisance_train_rounds,
            stream_seed=world_seed + DEFAULTS["NUISANCE_STREAM_SEED_OFFSET"],
            reward_std=reward_std,
            compact=True,
        )
        _set_model_seed(model_seed)
        frozen_q = fit_action_reward_model(
            nuisance_stream,
            random_state=model_seed,
            prediction_context=world["fixed_user_contexts"],
        )
        frozen_f = {}
        frozen_diagnostics = {}
        for partition, item in partition_sweep.items():
            _set_model_seed(model_seed)
            clusters_3d = clusters_to_onehot_3d(item["labels"], int(world["n_users"]))
            frozen_f[partition] = train_reward_model_via_two_stage(
                bandit_data=nuisance_stream,
                clusters=clusters_3d,
                need_q_x_a=False,
                random_state=model_seed,
                prediction_context=world["fixed_user_contexts"],
            )
            frozen_diagnostics[partition] = _local_correctness_diagnostics(
                world, frozen_f[partition], item["labels"]
            )
        frozen_models = (frozen_q, frozen_f, frozen_diagnostics)

    for stream_index in range(n_streams):
        full_stream = sample_logged_stream(
            world,
            n_rounds=max(n_list),
            stream_seed=world_seed + DEFAULTS["STREAM_SEED_OFFSET"] + stream_index,
            reward_std=reward_std,
            compact=compact_evaluation,
        )
        previous_prefixes = {}
        for n in n_list:
            prefix = build_feedback_prefix(full_stream, n)
            _assert_world_unchanged(prefix, world, world["true_value"], eps)
            _assert_nested_prefix(prefix, previous_prefixes)
            previous_prefixes[n] = prefix

            if "end_to_end" in arms:
                _set_model_seed(model_seed)
                q_population = fit_action_reward_model(
                    prefix,
                    random_state=model_seed,
                    prediction_context=world["fixed_user_contexts"],
                )
                f_populations = {}
                diagnostics = {}
                for partition, item in partition_sweep.items():
                    _set_model_seed(model_seed)
                    clusters_3d = clusters_to_onehot_3d(
                        item["labels"], int(world["n_users"])
                    )
                    f_populations[partition] = train_reward_model_via_two_stage(
                        bandit_data=prefix,
                        clusters=clusters_3d,
                        need_q_x_a=False,
                        random_state=model_seed,
                        prediction_context=world["fixed_user_contexts"],
                    )
                    diagnostics[partition] = _local_correctness_diagnostics(
                        world, f_populations[partition], item["labels"]
                    )
                _append_estimates(
                    rows, "end_to_end", prefix, world, stream_index, model_seed,
                    partition_sweep, q_population, f_populations, diagnostics,
                    compact_evaluation,
                )

            if frozen_models is not None:
                frozen_q, frozen_f, frozen_diagnostics = frozen_models
                _append_estimates(
                    rows, "frozen", prefix, world, stream_index, model_seed,
                    partition_sweep, frozen_q, frozen_f, frozen_diagnostics,
                    compact_evaluation,
                )
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f"world={world_index + 1} stream={stream_index + 1}/{n_streams} n={n}"
                )
    return rows


def _result_row(n, seed_index, estimator, estimate, true_value, **metadata):
    error = float(estimate - true_value)
    row = {
        "n": int(n),
        "seed": int(seed_index),
        "estimator": estimator,
        "estimate": float(estimate),
        "true_value": float(true_value),
        "error": error,
        "squared_error": float(error**2),
    }
    row.update(metadata)
    return row


def _assert_fixed_world(
    full_data: Dict,
    true_value: float,
    partitions: Dict[str, np.ndarray],
) -> None:
    if not np.isfinite(true_value):
        raise AssertionError("true value must be finite")
    expected = full_data["fixed_expected_rewards"]
    action_features = full_data["action_context_one_hot"]
    clusters = full_data["cluster_indices"]
    if expected.ndim != 2 or action_features.ndim != 2 or clusters.ndim != 1:
        raise AssertionError("fixed world arrays have invalid dimensions")
    for labels in partitions.values():
        if labels.shape != clusters.shape:
            raise AssertionError("partition labels must have one entry per action")


def _assert_world_unchanged(
    prefix: Dict,
    full_data: Dict,
    true_value: float,
    eps: float,
) -> None:
    np.testing.assert_array_equal(
        prefix["fixed_expected_rewards"],
        full_data["fixed_expected_rewards"],
    )
    np.testing.assert_array_equal(
        prefix["action_context_one_hot"],
        full_data["action_context_one_hot"],
    )
    np.testing.assert_array_equal(
        prefix["cluster_indices"],
        full_data["cluster_indices"],
    )
    check_value, _ = population_policy_value(prefix["fixed_expected_rewards"], eps)
    if not np.isclose(check_value, true_value):
        raise AssertionError("V_true changed across prefixes")


def _assert_nested_prefix(prefix: Dict, previous_prefixes: Dict[int, Dict]) -> None:
    for smaller_n, smaller in previous_prefixes.items():
        if smaller_n > prefix["n_rounds"]:
            continue
        for key in ROUND_LEVEL_FIELDS:
            if key not in prefix or prefix[key] is None:
                continue
            np.testing.assert_array_equal(prefix[key][:smaller_n], smaller[key])


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        arm = row.get("arm", "end_to_end")
        partition = row.get("partition")
        if partition is None:
            partition = "baseline" if row["estimator"] in {"DR", "DM"} else row["estimator"]
        grouped.setdefault((arm, row["n"], partition, row["estimator"]), []).append(row)

    aggregates = []
    for (arm, n, partition, estimator), items in sorted(grouped.items()):
        estimates = np.array([item["estimate"] for item in items], dtype=float)
        true_values = np.array([item["true_value"] for item in items], dtype=float)
        errors = estimates - true_values
        by_world = {}
        for item in items:
            world_seed = int(item.get("world_seed", item["seed"]))
            by_world.setdefault(world_seed, []).append(item)
        world_bias2 = []
        world_variances = []
        world_mses = []
        for world_items in by_world.values():
            world_errors = np.array(
                [item["estimate"] - item["true_value"] for item in world_items],
                dtype=float,
            )
            world_bias2.append(float(world_errors.mean() ** 2))
            world_variances.append(float(np.mean((world_errors - world_errors.mean()) ** 2)))
            world_mses.append(float(np.mean(world_errors**2)))

        aggregate = {
            "arm": arm,
            "n": int(n),
            "partition": partition,
            "estimator": estimator,
            "mse": float(np.mean(errors**2)),
            "rel_mse": float(np.mean((errors**2) / np.maximum(true_values**2, 1e-12))),
            "bias2": float(np.mean(world_bias2)),
            "variance": float(np.mean(world_variances)),
            "between_world_error_variance": float(np.var([np.mean([
                item["estimate"] - item["true_value"] for item in world_items
            ]) for world_items in by_world.values()])),
            "estimate_mean": float(estimates.mean()),
            "true_value_mean": float(true_values.mean()),
            "n_worlds": len(by_world),
            "n_stream_rows": len(items),
        }
        numeric_keys = set().union(*(item.keys() for item in items))
        excluded = {
            "n", "seed", "world_seed", "stream_seed", "stream_index", "model_seed",
            "arm", "partition", "estimator", "corruption_level", "estimate",
            "true_value", "error", "squared_error",
        }
        for key in sorted(numeric_keys - excluded):
            values = [item.get(key, np.nan) for item in items]
            try:
                values = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue
            if np.isfinite(values).any():
                aggregate[f"{key}_mean"] = float(np.nanmean(values))
        corruption_values = [item.get("corruption_level", np.nan) for item in items]
        if np.isfinite(np.asarray(corruption_values, dtype=float)).any():
            aggregate["corruption_level"] = float(np.nanmean(corruption_values))
        aggregates.append(aggregate)
    return aggregates


def _metric_from_items(items: List[Dict], key: str) -> float:
    estimates = np.array([item["estimate"] for item in items], dtype=float)
    true_values = np.array([item["true_value"] for item in items], dtype=float)
    errors = estimates - true_values
    if key == "mse":
        return float(np.mean(errors**2))
    if key == "rel_mse":
        norm = float(np.mean(true_values**2))
        return float(np.mean(errors**2) / max(norm, 1e-12))
    if key == "bias2":
        return float(np.mean(errors) ** 2)
    if key == "variance":
        return float(np.mean((errors - errors.mean()) ** 2))
    raise ValueError(f"unsupported metric for error bars: {key}")


def bootstrap_metric_se(
    items: List[Dict],
    key: str,
    n_bootstrap: int = 1000,
    random_state: int = 12345,
) -> float:
    if len(items) <= 1:
        return 0.0
    rng = np.random.RandomState(random_state)
    bootstrap_values = []
    n_items = len(items)
    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n_items, size=n_items)
        sample = [items[i] for i in sample_idx]
        bootstrap_values.append(_metric_from_items(sample, key))
    return float(np.std(bootstrap_values, ddof=1))


def plot_errorbars_by_group(rows: Iterable[Dict]) -> Dict[tuple, Dict[str, float]]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["n"], row["estimator"]), []).append(row)
    errorbars = {}
    for key, items in grouped.items():
        seed = int(key[0]) + sum(ord(ch) for ch in key[1])
        errorbars[key] = {
            metric: bootstrap_metric_se(
                items,
                metric,
                random_state=seed,
            )
            for metric in ("rel_mse", "bias2", "variance")
        }
    return errorbars


def write_csv(path: Path, rows: List[Dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_tidy_csv(path: Path) -> List[Dict]:
    import csv

    with open(path) as file:
        rows = []
        for raw in csv.DictReader(file):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = np.nan
                elif key in {"n", "seed", "world_seed", "stream_seed", "stream_index", "model_seed"}:
                    row[key] = int(value)
                elif key in {"estimator", "arm", "partition"}:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            row.setdefault("error", row["estimate"] - row["true_value"])
            rows.append(row)
        return rows


def plot_aggregates(
    aggregates: List[Dict],
    out_dir: Path,
    raw_rows: List[Dict] = None,
) -> None:
    metrics = [
        ("rel_mse", "Relative MSE", "rel_mse_vs_n.png"),
        ("bias2", "Bias^2", "bias2_vs_n.png"),
        ("variance", "Variance", "variance_vs_n.png"),
    ]
    estimators = [
        "OffCEM matched",
        "OffCEM reestimated WFSS",
        "OffCEM K-means",
        "DR",
        "DM",
    ]
    colors = {
        "OffCEM matched": "#1f77b4",
        "OffCEM reestimated WFSS": "#ff7f0e",
        "OffCEM K-means": "#2ca02c",
        "DR": "#d62728",
        "DM": "#9467bd",
    }
    arms = sorted({row.get("arm", "end_to_end") for row in aggregates})
    for arm in arms:
        arm_aggregates = [
            row for row in aggregates if row.get("arm", "end_to_end") == arm
        ]
        arm_rows = (
            [row for row in raw_rows if row.get("arm", "end_to_end") == arm]
            if raw_rows is not None
            else None
        )
        errorbars = plot_errorbars_by_group(arm_rows) if arm_rows is not None else {}
        arm_estimators = sorted({row["estimator"] for row in arm_aggregates})
        for key, ylabel, filename in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            for index, estimator in enumerate(arm_estimators):
                selected = [
                    row for row in arm_aggregates if row["estimator"] == estimator
                ]
                selected = sorted(selected, key=lambda row: row["n"])
                x = [row["n"] for row in selected]
                y = [row[key] for row in selected]
                yerr = [
                    errorbars.get((row["n"], estimator), {}).get(key, 0.0)
                    for row in selected
                ]
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr if arm_rows is not None else None,
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    capsize=3,
                    elinewidth=1.0,
                    label=estimator,
                    color=colors.get(estimator, plt.cm.tab20(index)),
                )
            ax.set_xscale("log")
            ax.set_xlabel("logged interactions n")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            target = out_dir / (filename if len(arms) == 1 else f"{arm}_{filename}")
            fig.savefig(target, dpi=150, bbox_inches="tight")
            plt.close(fig)

        lc_rows = [
            row for row in arm_aggregates if "lc_error_mean" in row and np.isfinite(row["lc_error_mean"])
        ]
        if lc_rows:
            fig, ax = plt.subplots(figsize=(7, 5))
            for n in sorted({row["n"] for row in lc_rows}):
                selected = [row for row in lc_rows if row["n"] == n]
                ax.scatter(
                    [row["lc_error_mean"] for row in selected],
                    [row["rel_mse"] for row in selected],
                    label=f"n={n}",
                )
            ax.set_xlabel("pi0-demeaned local-correctness error")
            ax.set_ylabel("Relative MSE")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            target = out_dir / (
                "rel_mse_vs_lc_error.png"
                if len(arms) == 1
                else f"{arm}_rel_mse_vs_lc_error.png"
            )
            fig.savefig(target, dpi=150, bbox_inches="tight")
            plt.close(fig)


def run_experiment(args) -> None:
    if args.quick:
        args.n_list = "500,3000"
        args.n_worlds = 2
        args.n_streams = 2
        args.partitions = "matched,corrupt"
        args.corruption_levels = "0,0.5"
        args.arms = "end_to_end"
        print("[quick] n-list=500,3000 n-worlds=2 n-streams=2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "sample_size_stress_tidy.csv"
    aggregate_path = out_dir / "sample_size_stress_aggregate.csv"

    if args.plot_only:
        rows = load_tidy_csv(tidy_path)
        aggregates = aggregate_results(rows)
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir, raw_rows=rows)
        return

    n_list = parse_n_list(args.n_list)
    n_worlds = args.n_seeds if args.n_worlds is None else args.n_worlds
    if n_worlds <= 0 or args.n_streams <= 0:
        raise ValueError("--n-worlds and --n-streams must be positive")
    partition_names = parse_name_list(args.partitions, "partitions")
    corruption_levels = parse_float_list(args.corruption_levels, "corruption-levels")
    arms = parse_name_list(args.arms, "arms")
    if set(arms) - {"end_to_end", "frozen"}:
        raise ValueError("--arms supports only end_to_end,frozen")
    if args.nuisance_train_rounds <= 0:
        raise ValueError("--nuisance-train-rounds must be positive")
    if max(n_list) > 100000 and not args.compact_evaluation:
        raise ValueError(
            "sample sizes above 100000 require --compact-evaluation to avoid "
            "dense round-by-action arrays"
        )
    all_rows = []
    started = time()
    total_prefixes = n_worlds * args.n_streams * len(n_list)
    progress = tqdm(total=total_prefixes, desc="sample-size stream prefixes")
    for world_index in range(n_worlds):
        progress.set_postfix_str(f"world={world_index + 1} initializing")
        all_rows.extend(
            run_world(
                world_index=world_index,
                n_list=n_list,
                n_clusters=args.n_clusters,
                eps=args.eps,
                reward_std=args.reward_std,
                n_streams=args.n_streams,
                partition_names=partition_names,
                corruption_levels=corruption_levels,
                arms=arms,
                nuisance_train_rounds=args.nuisance_train_rounds,
                base_model_seed=args.base_model_seed,
                compact_evaluation=args.compact_evaluation,
                progress=progress,
            )
        )
        with open(out_dir / "latest_rows.json", "w") as file:
            json.dump(all_rows, file, indent=2)
    progress.close()

    aggregates = aggregate_results(all_rows)
    write_csv(tidy_path, all_rows)
    write_csv(aggregate_path, aggregates)
    if not args.no_plot:
        plot_aggregates(aggregates, out_dir, raw_rows=all_rows)
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
