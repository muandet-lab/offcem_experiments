"""Synthetic domain-informed and outcome-aware clustering experiment."""
from dataclasses import dataclass
from typing import Dict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
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
    return centered / max(centered.std(), EPS)


@dataclass
class DomainOutcomeDataset:
    n_actions: int
    n_users: int
    n_clusters: int
    dim_context: int
    dim_generic: int
    dim_domain: int
    domain_alignment: float
    domain_label_noise: float
    action_noise: float
    reward_scale: float
    reward_std: float
    beta: float
    target_epsilon: float
    random_state: int

    def generate(self, n_rounds: int, sample_random_state: int) -> Dict:
        if not 0 <= self.domain_alignment <= 1:
            raise ValueError("domain_alignment must be in [0, 1]")
        if not 0 <= self.domain_label_noise <= 1:
            raise ValueError("domain_label_noise must be in [0, 1]")

        environment = self._build_environment()
        sample_rng = check_random_state(sample_random_state)
        user_idx = sample_rng.choice(self.n_users, size=n_rounds)
        pi_b_rows = environment["behavior_policy"][user_idx]
        actions = np.array(
            [
                sample_rng.choice(self.n_actions, p=probability)
                for probability in pi_b_rows
            ],
            dtype=int,
        )
        expected_reward = environment["q_population"][user_idx]
        factual_mean = expected_reward[np.arange(n_rounds), actions]
        rewards = sample_rng.normal(factual_mean, self.reward_std)

        reward_mat = np.zeros((self.n_users, self.n_actions))
        obs_mat = np.zeros((self.n_users, self.n_actions), dtype=int)
        for user, action, reward in zip(user_idx, actions, rewards):
            reward_mat[user, action] = reward
            obs_mat[user, action] = 1

        generic_labels = environment["generic_labels"]
        p_e_a = np.zeros(
            (self.n_actions, self.n_clusters, 1),
            dtype=float,
        )
        p_e_a[np.arange(self.n_actions), generic_labels, 0] = 1.0
        return {
            "n_rounds": n_rounds,
            "n_users": self.n_users,
            "n_actions": self.n_actions,
            "context": environment["contexts"][user_idx],
            "fixed_user_contexts": environment["contexts"],
            "user_idx": user_idx,
            "action": actions,
            "reward": rewards,
            "position": None,
            "pscore": pi_b_rows[np.arange(n_rounds), actions],
            "pi_b": pi_b_rows[:, :, np.newaxis],
            "pi_b_population": environment["behavior_policy"],
            "target_policy_population": environment["target_policy"],
            "expected_reward": expected_reward,
            "fixed_expected_rewards": environment["q_population"],
            "action_context": environment["generic_features"],
            "action_context_one_hot": environment["generic_features"],
            "action_embed": generic_labels[actions, np.newaxis],
            "p_e_a": p_e_a,
            "cluster_indices": environment["domain_labels"],
            "clusters": clusters_to_onehot_3d(
                environment["domain_labels"], self.n_users
            ),
            "reward_mat": reward_mat,
            "obs_mat": obs_mat,
            "support_mask": np.ones(self.n_actions, dtype=bool),
            "generic_features": environment["generic_features"],
            "domain_features": environment["domain_features"],
            "generic_labels": generic_labels,
            "domain_labels": environment["domain_labels"],
            "observed_domain_labels": environment[
                "observed_domain_labels"
            ],
            "outcome_labels": environment["outcome_labels"],
            "domain_component": environment["domain_component"],
            "outcome_component": environment["outcome_component"],
            "true_policy_value": float(
                (
                    environment["target_policy"]
                    * environment["q_population"]
                )
                .sum(axis=1)
                .mean()
            ),
        }

    def _build_environment(self) -> Dict:
        rng = check_random_state(self.random_state)
        contexts = rng.normal(size=(self.n_users, self.dim_context))
        generic_labels = _balanced_labels(
            self.n_actions, self.n_clusters, rng
        )
        domain_labels = _balanced_labels(
            self.n_actions, self.n_clusters, rng
        )
        outcome_labels = _balanced_labels(
            self.n_actions, self.n_clusters, rng
        )

        generic_centers = rng.normal(
            size=(self.n_clusters, self.dim_generic)
        )
        domain_centers = rng.normal(
            size=(self.n_clusters, self.dim_domain)
        )
        generic_features = generic_centers[generic_labels]
        generic_features += rng.normal(
            scale=self.action_noise,
            size=generic_features.shape,
        )

        observed_domain_labels = domain_labels.copy()
        n_corrupt = int(round(self.domain_label_noise * self.n_actions))
        if n_corrupt:
            corrupted = rng.choice(
                self.n_actions, size=n_corrupt, replace=False
            )
            offsets = rng.randint(1, self.n_clusters, size=n_corrupt)
            observed_domain_labels[corrupted] = (
                observed_domain_labels[corrupted] + offsets
            ) % self.n_clusters
        domain_features = domain_centers[observed_domain_labels]
        domain_features += rng.normal(
            scale=self.action_noise,
            size=domain_features.shape,
        )

        domain_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_clusters),
        )
        outcome_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_clusters),
        )
        action_deviation = rng.normal(
            scale=0.2 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_actions),
        )
        domain_component = (contexts @ domain_w)[:, domain_labels]
        outcome_component = (
            (contexts @ outcome_w)[:, outcome_labels]
            + contexts @ action_deviation
        )
        q_population = (
            np.sqrt(self.domain_alignment)
            * _standardize(domain_component)
            + np.sqrt(1.0 - self.domain_alignment)
            * _standardize(outcome_component)
        )
        q_population *= self.reward_scale / max(q_population.std(), EPS)

        target_policy = np.full_like(
            q_population, self.target_epsilon / self.n_actions
        )
        target_policy[
            np.arange(self.n_users), q_population.argmax(axis=1)
        ] += 1.0 - self.target_epsilon
        behavior_policy = _softmax(self.beta * q_population)
        return {
            "contexts": contexts,
            "generic_labels": generic_labels,
            "domain_labels": domain_labels,
            "observed_domain_labels": observed_domain_labels,
            "outcome_labels": outcome_labels,
            "generic_features": _scale_features(generic_features),
            "domain_features": _scale_features(domain_features),
            "domain_component": domain_component,
            "outcome_component": outcome_component,
            "q_population": q_population,
            "target_policy": target_policy,
            "behavior_policy": behavior_policy,
        }


def build_partitions(
    evaluation_data: Dict,
    auxiliary_q: np.ndarray,
    n_clusters: int,
    random_state: int,
    profile_components: int,
) -> Dict[str, np.ndarray]:
    generic = evaluation_data["generic_features"]
    domain = evaluation_data["domain_features"]
    oracle_profile = evaluation_data["fixed_expected_rewards"].T
    estimated_profile = auxiliary_q[:, :, 0].T
    oracle_embedding = profile_embedding(
        oracle_profile, profile_components, random_state
    )
    estimated_embedding = profile_embedding(
        estimated_profile, profile_components, random_state
    )
    hybrid = np.concatenate(
        [
            _scale_features(domain),
            _scale_features(estimated_embedding),
        ],
        axis=1,
    )
    rng = check_random_state(random_state)
    shuffled_domain = domain[rng.permutation(len(domain))]
    return {
        "feature_only": _kmeans(generic, n_clusters, random_state),
        "domain_informed": _kmeans(domain, n_clusters, random_state),
        "oracle_outcome": _kmeans(
            oracle_embedding, n_clusters, random_state
        ),
        "estimated_outcome": _kmeans(
            estimated_embedding, n_clusters, random_state
        ),
        "hybrid": _kmeans(hybrid, n_clusters, random_state),
        "shuffled_domain": _kmeans(
            shuffled_domain, n_clusters, random_state
        ),
        "random": _balanced_labels(
            len(generic), n_clusters, rng
        ),
    }


def profile_embedding(
    profiles: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    scaled = _scale_features(profiles)
    components = min(
        n_components,
        scaled.shape[0] - 1,
        scaled.shape[1],
    )
    if components < 1:
        return scaled
    return PCA(
        n_components=components,
        random_state=random_state,
    ).fit_transform(scaled)


def partition_diagnostics(
    labels: np.ndarray,
    data: Dict,
) -> Dict[str, float]:
    sizes = np.bincount(
        labels, minlength=int(labels.max()) + 1
    )
    return {
        "ari_domain": float(
            adjusted_rand_score(data["domain_labels"], labels)
        ),
        "ari_outcome": float(
            adjusted_rand_score(data["outcome_labels"], labels)
        ),
        "ari_generic": float(
            adjusted_rand_score(data["generic_labels"], labels)
        ),
        "cluster_size_cv": float(
            sizes.std() / max(sizes.mean(), EPS)
        ),
    }


def summarize_domain_outcome(
    records: Iterable[Dict],
    output_dir,
) -> Dict:
    method_rows = []
    baseline_rows = []
    for record in records:
        if "error" in record:
            continue
        truth = record["true_policy_value"]
        baseline_errors = {
            name: ((estimate - truth) / truth) ** 2
            for name, estimate in record["baselines"].items()
        }
        best_baseline = min(baseline_errors, key=baseline_errors.get)
        for name, estimate in record["baselines"].items():
            baseline_rows.append(
                {
                    "domain_alignment": record["domain_alignment"],
                    "domain_label_noise": record["domain_label_noise"],
                    "seed": record["seed"],
                    "estimator": name,
                    "estimate": estimate,
                    "relative_squared_error": baseline_errors[name],
                }
            )
        for method in record["methods"]:
            if "error" in method:
                continue
            relative_error = (method["estimate"] - truth) / truth
            method_rows.append(
                {
                    "domain_alignment": record["domain_alignment"],
                    "domain_label_noise": record["domain_label_noise"],
                    "seed": record["seed"],
                    "method": method["method"],
                    "estimate": method["estimate"],
                    "relative_error": relative_error,
                    "relative_squared_error": relative_error**2,
                    "best_baseline": best_baseline,
                    "best_baseline_error": baseline_errors[best_baseline],
                    "beats_best_baseline": (
                        relative_error**2
                        < baseline_errors[best_baseline]
                    ),
                    **method["diagnostics"],
                    **method["partition_diagnostics"],
                }
            )

    methods = pd.DataFrame(method_rows)
    baselines = pd.DataFrame(baseline_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods.to_csv(output_dir / "method_results.csv", index=False)
    baselines.to_csv(output_dir / "baseline_results.csv", index=False)
    method_summary = _summarize_methods(methods)
    baseline_summary = _summarize_baselines(baselines)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    baseline_summary.to_csv(
        output_dir / "baseline_summary.csv", index=False
    )
    return {
        "n_method_rows": len(methods),
        "n_baseline_rows": len(baselines),
        "methods": method_summary.to_dict(orient="records"),
        "baselines": baseline_summary.to_dict(orient="records"),
    }


def _summarize_methods(methods: pd.DataFrame) -> pd.DataFrame:
    if methods.empty:
        return pd.DataFrame()
    grouped = methods.groupby(
        ["domain_alignment", "domain_label_noise", "method"]
    )
    rows = []
    for keys, group in grouped:
        normalized = group["relative_error"].to_numpy()
        rows.append(
            {
                "domain_alignment": keys[0],
                "domain_label_noise": keys[1],
                "method": keys[2],
                "n_seeds": len(group),
                "relMSE": float(np.mean(normalized**2)),
                "relBias2": float(np.mean(normalized) ** 2),
                "relVar": float(np.var(normalized)),
                "win_rate_best_baseline": float(
                    group["beats_best_baseline"].mean()
                ),
                "dm_2s_mean": float(group["dm_2s"].mean()),
                "dm_policy_weighted_mean": float(
                    group["dm_policy_weighted"].mean()
                ),
                "ari_domain_mean": float(group["ari_domain"].mean()),
                "ari_outcome_mean": float(group["ari_outcome"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_baselines(baselines: pd.DataFrame) -> pd.DataFrame:
    if baselines.empty:
        return pd.DataFrame()
    return (
        baselines.groupby(
            ["domain_alignment", "domain_label_noise", "estimator"]
        )["relative_squared_error"]
        .mean()
        .rename("relMSE")
        .reset_index()
    )


def _balanced_labels(n_actions, n_clusters, rng):
    labels = np.arange(n_actions) % n_clusters
    rng.shuffle(labels)
    return labels.astype(int)


def _scale_features(features):
    return StandardScaler().fit_transform(np.asarray(features, dtype=float))


def _kmeans(features, n_clusters, random_state):
    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit_predict(features)
