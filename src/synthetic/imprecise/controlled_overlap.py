"""Controlled FCM experiment with known reward-relevant memberships."""
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from clustering import clusters_to_onehot_3d


EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _standardize(component: np.ndarray) -> np.ndarray:
    centered = component - component.mean()
    scale = centered.std()
    return centered / max(scale, EPS)


@dataclass
class ControlledOverlapDataset:
    n_actions: int
    n_users: int
    n_clusters: int
    dim_context: int
    dim_action: int
    ambiguity_fraction: float
    ambiguity_strength: float
    feature_noise: float
    reward_mode: str
    cluster_reward_share: float
    reward_scale: float
    reward_std: float
    beta: float
    target_epsilon: float
    random_state: int

    def generate(self, n_rounds: int, sample_random_state: int) -> Dict:
        if self.reward_mode not in {"reward_relevant", "feature_only"}:
            raise ValueError(
                "reward_mode must be reward_relevant or feature_only"
            )
        if not 0 <= self.ambiguity_fraction <= 1:
            raise ValueError("ambiguity_fraction must be in [0, 1]")
        if not 0 <= self.ambiguity_strength <= 0.5:
            raise ValueError("ambiguity_strength must be in [0, 0.5]")
        if not 0 <= self.cluster_reward_share <= 1:
            raise ValueError("cluster_reward_share must be in [0, 1]")

        rng = check_random_state(self.random_state)
        sample_rng = check_random_state(sample_random_state)
        contexts = rng.normal(size=(self.n_users, self.dim_context))
        centers = rng.normal(size=(self.n_clusters, self.dim_action))
        centers /= np.maximum(
            np.linalg.norm(centers, axis=1, keepdims=True), EPS
        )

        primary = np.arange(self.n_actions) % self.n_clusters
        rng.shuffle(primary)
        secondary = np.empty(self.n_actions, dtype=int)
        for action, cluster in enumerate(primary):
            choices = np.delete(np.arange(self.n_clusters), cluster)
            secondary[action] = rng.choice(choices)

        ambiguous = np.zeros(self.n_actions, dtype=bool)
        n_ambiguous = int(round(self.ambiguity_fraction * self.n_actions))
        if n_ambiguous:
            ambiguous[
                rng.choice(self.n_actions, n_ambiguous, replace=False)
            ] = True
        secondary_weight = np.zeros(self.n_actions)
        if n_ambiguous:
            secondary_weight[ambiguous] = rng.uniform(
                0.5 * self.ambiguity_strength,
                self.ambiguity_strength,
                size=n_ambiguous,
            )
        memberships = np.zeros((self.n_actions, self.n_clusters))
        memberships[np.arange(self.n_actions), primary] = (
            1.0 - secondary_weight
        )
        memberships[np.arange(self.n_actions), secondary] += secondary_weight

        action_features = memberships @ centers
        action_features += rng.normal(
            scale=self.feature_noise,
            size=action_features.shape,
        )
        action_features = (
            action_features - action_features.mean(axis=0, keepdims=True)
        )
        action_features /= np.maximum(
            action_features.std(axis=0, keepdims=True), EPS
        )

        cluster_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_clusters),
        )
        cluster_profiles = contexts @ cluster_w
        primary_component = cluster_profiles[:, primary]
        soft_component = cluster_profiles @ memberships.T
        cluster_component = (
            soft_component
            if self.reward_mode == "reward_relevant"
            else primary_component
        )
        identity_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_actions),
        )
        identity_component = contexts @ identity_w
        q_population = (
            np.sqrt(self.cluster_reward_share)
            * _standardize(cluster_component)
            + np.sqrt(1.0 - self.cluster_reward_share)
            * _standardize(identity_component)
        )
        q_population *= self.reward_scale / max(q_population.std(), EPS)

        target_policy = np.full_like(
            q_population, self.target_epsilon / self.n_actions
        )
        target_policy[
            np.arange(self.n_users), q_population.argmax(axis=1)
        ] += 1.0 - self.target_epsilon
        behavior_policy = _softmax(self.beta * q_population)

        user_idx = sample_rng.choice(self.n_users, size=n_rounds)
        pi_b_rows = behavior_policy[user_idx]
        actions = np.array(
            [
                sample_rng.choice(self.n_actions, p=probability)
                for probability in pi_b_rows
            ],
            dtype=int,
        )
        expected_reward = q_population[user_idx]
        factual_mean = expected_reward[np.arange(n_rounds), actions]
        rewards = sample_rng.normal(factual_mean, self.reward_std)

        reward_mat = np.zeros((self.n_users, self.n_actions))
        obs_mat = np.zeros((self.n_users, self.n_actions), dtype=int)
        for user, action, reward in zip(user_idx, actions, rewards):
            reward_mat[user, action] = reward
            obs_mat[user, action] = 1

        p_e_a = np.eye(self.n_actions)[:, :, np.newaxis]
        reward_mixture_effect = np.sqrt(
            np.mean((soft_component - primary_component) ** 2, axis=0)
        )
        if self.reward_mode == "feature_only":
            reward_mixture_effect = np.zeros_like(reward_mixture_effect)

        return {
            "n_rounds": n_rounds,
            "n_users": self.n_users,
            "n_actions": self.n_actions,
            "context": contexts[user_idx],
            "fixed_user_contexts": contexts,
            "user_idx": user_idx,
            "action": actions,
            "reward": rewards,
            "position": None,
            "pscore": pi_b_rows[np.arange(n_rounds), actions],
            "pi_b": pi_b_rows[:, :, np.newaxis],
            "pi_b_population": behavior_policy,
            "target_policy_population": target_policy,
            "expected_reward": expected_reward,
            "fixed_expected_rewards": q_population,
            "action_context": action_features,
            "action_context_one_hot": action_features,
            "action_embed": actions[:, np.newaxis],
            "p_e_a": p_e_a,
            "cluster_indices": primary,
            "clusters": clusters_to_onehot_3d(primary, self.n_users),
            "reward_mat": reward_mat,
            "obs_mat": obs_mat,
            "support_mask": np.ones(self.n_actions, dtype=bool),
            "true_memberships": memberships,
            "primary_clusters": primary,
            "secondary_clusters": secondary,
            "ambiguous_actions": ambiguous,
            "secondary_weight": secondary_weight,
            "latent_centers": centers,
            "reward_mixture_effect": reward_mixture_effect,
            "true_policy_value": float(
                (target_policy * q_population).sum(axis=1).mean()
            ),
        }


def align_memberships(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Permute estimated membership columns to best match known memberships."""
    similarity = estimated.T @ truth
    estimated_columns, truth_columns = linear_sum_assignment(-similarity)
    aligned = np.zeros_like(estimated)
    aligned[:, truth_columns] = estimated[:, estimated_columns]
    return aligned


def membership_diagnostics(
    estimated: np.ndarray,
    truth: np.ndarray,
    reward_mixture_effect: np.ndarray,
) -> Dict[str, float]:
    aligned = align_memberships(estimated, truth)
    true_entropy = normalized_entropy(truth)
    estimated_entropy = normalized_entropy(aligned)
    true_labels = truth.argmax(axis=1)
    estimated_labels = aligned.argmax(axis=1)
    ambiguous = true_entropy > 1e-8
    result = {
        "membership_mse": float(np.mean((aligned - truth) ** 2)),
        "primary_accuracy": float(np.mean(estimated_labels == true_labels)),
        "true_entropy_mean": float(true_entropy.mean()),
        "estimated_entropy_mean": float(estimated_entropy.mean()),
        "entropy_spearman": _safe_spearman(
            true_entropy, estimated_entropy
        ),
        "reward_relevance_spearman": _safe_spearman(
            estimated_entropy, reward_mixture_effect
        ),
        "reward_mixture_effect_mean": float(reward_mixture_effect.mean()),
    }
    if ambiguous.any() and (~ambiguous).any():
        result["ambiguity_auc"] = float(
            roc_auc_score(ambiguous.astype(int), estimated_entropy)
        )
    else:
        result["ambiguity_auc"] = np.nan
    return result


def normalized_entropy(memberships: np.ndarray) -> np.ndarray:
    probabilities = np.maximum(memberships, EPS)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropy / max(np.log(memberships.shape[1]), EPS)


def cluster_size_diagnostics(labels: np.ndarray, n_clusters: int) -> Dict:
    sizes = np.bincount(labels, minlength=n_clusters)
    return {
        "cluster_size_min": int(sizes.min()),
        "cluster_size_max": int(sizes.max()),
        "cluster_size_cv": float(
            sizes.std() / max(sizes.mean(), EPS)
        ),
    }


def summarize_controlled_records(
    records: Iterable[Dict],
    output_dir,
) -> Dict:
    records = [record for record in records if "error" not in record]
    fit_rows = []
    partition_rows = []
    for record in records:
        truth = record["true_policy_value"]
        baseline_errors = {
            name: ((estimate - truth) / truth) ** 2
            for name, estimate in record["baselines"].items()
        }
        best_baseline_error = min(baseline_errors.values())
        partition_lookup = {}
        for partition in record["partitions"]:
            if "error" in partition:
                continue
            relative_error = (partition["estimate"] - truth) / truth
            row = {
                "reward_mode": record["reward_mode"],
                "ambiguity_fraction": record["ambiguity_fraction"],
                "seed": record["seed"],
                "source": partition["source"],
                "draw": partition["draw"],
                "estimate": partition["estimate"],
                "relative_squared_error": relative_error**2,
                "dm_2s": partition["diagnostics"]["dm_2s"],
                "dm_ratio": partition["diagnostics"]["dm_ratio"],
                "dm_policy_weighted": partition["diagnostics"][
                    "dm_policy_weighted"
                ],
                "best_baseline_error": best_baseline_error,
                "loses_to_best_baseline": relative_error**2
                > best_baseline_error,
                **partition["partition_diagnostics"],
            }
            partition_rows.append(row)
            partition_lookup.setdefault(partition["source"], []).append(row)

        hard = partition_lookup["fcm_hard"][0]
        samples = partition_lookup.get("fcm_sample", [])
        sample_errors = np.array(
            [sample["relative_squared_error"] for sample in samples]
        )
        sample_estimates = np.array(
            [sample["estimate"] for sample in samples]
        )
        sample_dm = np.array(
            [sample["dm_policy_weighted"] for sample in samples]
        )
        fit_rows.append(
            {
                "reward_mode": record["reward_mode"],
                "ambiguity_fraction": record["ambiguity_fraction"],
                "seed": record["seed"],
                "true_policy_value": truth,
                **record["membership_diagnostics"],
                **record["fcm_uncertainty"],
                "hard_estimate": hard["estimate"],
                "hard_relMSE": hard["relative_squared_error"],
                "hard_dm_2s": hard["dm_2s"],
                "hard_dm_ratio": hard["dm_ratio"],
                "hard_dm_policy_weighted": hard["dm_policy_weighted"],
                "hard_ari_primary": hard["ari_primary"],
                "hard_cluster_size_cv": hard["cluster_size_cv"],
                "best_baseline_error": best_baseline_error,
                "hard_excess_error": (
                    hard["relative_squared_error"] - best_baseline_error
                ),
                "hard_loses_to_best_baseline": (
                    hard["relative_squared_error"] > best_baseline_error
                ),
                "sample_relMSE_mean": _mean_or_nan(sample_errors),
                "sample_relMSE_best": _min_or_nan(sample_errors),
                "sample_better_fraction": (
                    float(np.mean(sample_errors < hard["relative_squared_error"]))
                    if sample_errors.size
                    else np.nan
                ),
                "sample_estimate_range": (
                    float(sample_estimates.max() - sample_estimates.min())
                    if sample_estimates.size
                    else np.nan
                ),
                "sample_policy_dm_range": (
                    float(sample_dm.max() - sample_dm.min())
                    if sample_dm.size
                    else np.nan
                ),
                **{
                    f"{name}_error": error
                    for name, error in baseline_errors.items()
                },
            }
        )

    fits = pd.DataFrame(fit_rows)
    partitions = pd.DataFrame(partition_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output_dir / "fit_summary.csv", index=False)
    partitions.to_csv(output_dir / "partition_results.csv", index=False)
    aggregate = _aggregate_fits(fits)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    prediction = incremental_prediction_analysis(fits)
    prediction.to_csv(output_dir / "prediction_metrics.csv", index=False)
    return {
        "n_fits": len(fits),
        "n_partitions": len(partitions),
        "aggregate": aggregate.to_dict(orient="records"),
        "prediction": prediction.to_dict(orient="records"),
    }


def incremental_prediction_analysis(fits: pd.DataFrame) -> pd.DataFrame:
    """Held-overlap prediction of hard OffCEM error."""
    if fits.empty or fits["ambiguity_fraction"].nunique() < 2:
        return pd.DataFrame()
    feature_sets = {
        "policy_dm_only": ["hard_dm_policy_weighted"],
        "policy_dm_partition": [
            "hard_dm_policy_weighted",
            "hard_cluster_size_cv",
        ],
        "policy_dm_partition_uncertainty": [
            "hard_dm_policy_weighted",
            "hard_cluster_size_cv",
            "estimated_entropy_mean",
            "entropy_mean",
            "margin_mean",
            "ambiguous_fraction_margin_lt_0.1",
        ],
    }
    rows = []
    groups = fits["ambiguity_fraction"]
    target = fits["hard_relMSE"].to_numpy()
    for name, columns in feature_sets.items():
        predictions = np.full(len(fits), np.nan)
        for held_out in sorted(groups.unique()):
            train = groups != held_out
            test = groups == held_out
            if train.sum() < 2 or test.sum() == 0:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(fits.loc[train, columns], target[train])
            predictions[test] = model.predict(fits.loc[test, columns])
        valid = np.isfinite(predictions)
        if not valid.any():
            continue
        rows.append(
            {
                "model": name,
                "n_predictions": int(valid.sum()),
                "rmse": float(
                    np.sqrt(mean_squared_error(target[valid], predictions[valid]))
                ),
                "mae": float(
                    mean_absolute_error(target[valid], predictions[valid])
                ),
                "spearman": _safe_spearman(
                    target[valid], predictions[valid]
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_fits(fits: pd.DataFrame) -> pd.DataFrame:
    if fits.empty:
        return pd.DataFrame()
    columns = [
        "membership_mse",
        "primary_accuracy",
        "true_entropy_mean",
        "estimated_entropy_mean",
        "entropy_spearman",
        "ambiguity_auc",
        "reward_relevance_spearman",
        "reward_mixture_effect_mean",
        "hard_relMSE",
        "hard_dm_policy_weighted",
        "hard_excess_error",
        "hard_loses_to_best_baseline",
        "sample_relMSE_mean",
        "sample_relMSE_best",
        "sample_better_fraction",
        "sample_estimate_range",
    ]
    return (
        fits.groupby(["reward_mode", "ambiguity_fraction"])[columns]
        .mean()
        .reset_index()
    )


def _safe_spearman(first, second) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.size < 2 or np.std(first) <= EPS or np.std(second) <= EPS:
        return np.nan
    result = spearmanr(first, second)
    value = getattr(result, "statistic", result.correlation)
    return float(value)


def _mean_or_nan(values):
    return float(values.mean()) if values.size else np.nan


def _min_or_nan(values):
    return float(values.min()) if values.size else np.nan
