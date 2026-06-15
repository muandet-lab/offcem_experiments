"""Synthetic data generator for the OffCEM applicability benchmark."""
from dataclasses import dataclass
from typing import Dict
from typing import Sequence

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from clustering import compute_clusters
from clustering import clusters_to_onehot_3d


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", sparse=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _standardize_component(component: np.ndarray) -> np.ndarray:
    centered = component - component.mean()
    std = centered.std()
    if std < 1e-12:
        return centered
    return centered / std


def _validate_simplex(alpha: Sequence[float]) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (4,) or np.any(alpha < 0):
        raise ValueError("reward.alpha must contain four non-negative values")
    total = alpha.sum()
    if total <= 0:
        raise ValueError("reward.alpha must have positive mass")
    return alpha / total


@dataclass
class ApplicabilityDataset:
    """Generate one finite contextual-bandit population and one logged sample.

    The expected reward is decomposed into four standardized components:
    between-cluster, observed-feature, action-identity, and hidden-feature
    structure. ``sqrt(alpha)`` weights make ``alpha`` approximate variance
    shares when the components are weakly correlated. Realized shares are
    recorded because finite populations are not exactly orthogonal.
    """

    n_actions: int
    n_users: int
    dim_context: int
    n_cat_dim: int
    n_cat_per_dim: int
    n_unobserved_cat_dim: int
    n_clusters: int
    gen_clustering_method: str
    cluster_balance: str
    cluster_temperature: float
    alpha: Sequence[float]
    feature_nonlinearity: float
    reward_scale: float
    reward_std: float
    beta: float
    target_epsilon: float
    deficient_action_fraction: float
    random_state: int

    def generate(
        self,
        n_rounds: int,
        sample_random_state: int = None,
    ) -> Dict:
        environment_rng = check_random_state(self.random_state)
        sample_rng = check_random_state(
            self.random_state
            if sample_random_state is None
            else sample_random_state
        )
        alpha = _validate_simplex(self.alpha)
        if not 0 <= self.n_unobserved_cat_dim < self.n_cat_dim:
            raise ValueError("n_unobserved_cat_dim must be in [0, n_cat_dim)")
        if not 0 <= self.deficient_action_fraction < 1:
            raise ValueError("deficient_action_fraction must be in [0, 1)")

        contexts = environment_rng.normal(size=(self.n_users, self.dim_context))
        hidden_contexts = environment_rng.normal(
            size=(self.n_users, self.dim_context)
        )
        action_embed = environment_rng.randint(
            self.n_cat_per_dim, size=(self.n_actions, self.n_cat_dim)
        )
        observed_embed = action_embed[:, self.n_unobserved_cat_dim :]
        hidden_embed = action_embed[:, : self.n_unobserved_cat_dim]
        observed_one_hot = _one_hot_encoder().fit_transform(observed_embed)

        clusters = compute_clusters(
            action_features=observed_one_hot,
            n_clusters=self.n_clusters,
            method=self.gen_clustering_method,
            balance=self.cluster_balance,
            random_state=self.random_state,
            temperature=self.cluster_temperature,
        )
        n_clusters = int(clusters.max()) + 1

        components = self._reward_components(
            contexts=contexts,
            hidden_contexts=hidden_contexts,
            observed_features=observed_one_hot,
            hidden_embed=hidden_embed,
            clusters=clusters,
            n_clusters=n_clusters,
            rng=environment_rng,
        )
        weighted = [
            np.sqrt(alpha[i]) * components[name]
            for i, name in enumerate(("cluster", "feature", "identity", "hidden"))
        ]
        q_population = np.sum(weighted, axis=0)
        q_population *= self.reward_scale / max(q_population.std(), 1e-12)

        target_policy = np.full_like(
            q_population, self.target_epsilon / self.n_actions
        )
        target_policy[
            np.arange(self.n_users), q_population.argmax(axis=1)
        ] += 1.0 - self.target_epsilon

        behavior_full = _softmax(self.beta * q_population)
        n_deficient = int(round(self.deficient_action_fraction * self.n_actions))
        support_mask = np.ones(self.n_actions, dtype=bool)
        if n_deficient:
            deficient = check_random_state(self.random_state + 7001).choice(
                self.n_actions, size=n_deficient, replace=False
            )
            support_mask[deficient] = False
        behavior_policy = behavior_full * support_mask[np.newaxis, :]
        behavior_policy /= behavior_policy.sum(axis=1, keepdims=True)

        user_idx = sample_rng.choice(self.n_users, size=n_rounds)
        pi_b_rows = behavior_policy[user_idx]
        actions = np.array(
            [sample_rng.choice(self.n_actions, p=p) for p in pi_b_rows],
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

        p_e_a = np.zeros(
            (self.n_actions, self.n_cat_per_dim, observed_embed.shape[1])
        )
        for d in range(observed_embed.shape[1]):
            p_e_a[np.arange(self.n_actions), observed_embed[:, d], d] = 1.0

        total_flat = np.sum(weighted, axis=0).ravel()
        total_centered = total_flat - total_flat.mean()
        total_variance = float(np.mean(total_centered**2))
        realized_shares = {}
        for name, term in zip(
            ("cluster", "feature", "identity", "hidden"), weighted
        ):
            term_flat = term.ravel()
            covariance = float(
                np.mean((term_flat - term_flat.mean()) * total_centered)
            )
            realized_shares[name] = (
                covariance / total_variance if total_variance else 0.0
            )

        return dict(
            n_rounds=n_rounds,
            n_users=self.n_users,
            n_actions=self.n_actions,
            context=contexts[user_idx],
            fixed_user_contexts=contexts,
            hidden_user_contexts=hidden_contexts,
            user_idx=user_idx,
            action=actions,
            reward=rewards,
            position=None,
            pscore=pi_b_rows[np.arange(n_rounds), actions],
            pi_b=pi_b_rows[:, :, np.newaxis],
            pi_b_population=behavior_policy,
            pi_b_full_population=behavior_full,
            target_policy_population=target_policy,
            expected_reward=expected_reward,
            fixed_expected_rewards=q_population,
            action_context=observed_embed,
            action_context_one_hot=observed_one_hot,
            action_embed=observed_embed[actions],
            p_e_a=p_e_a,
            cluster_indices=clusters,
            clusters=clusters_to_onehot_3d(clusters, self.n_users),
            reward_mat=reward_mat,
            obs_mat=obs_mat,
            support_mask=support_mask,
            reward_components=components,
            nominal_variance_shares=dict(
                zip(("cluster", "feature", "identity", "hidden"), alpha.tolist())
            ),
            realized_variance_shares=realized_shares,
            true_policy_value=float((target_policy * q_population).sum(1).mean()),
        )

    def _reward_components(
        self,
        contexts: np.ndarray,
        hidden_contexts: np.ndarray,
        observed_features: np.ndarray,
        hidden_embed: np.ndarray,
        clusters: np.ndarray,
        n_clusters: int,
        rng: np.random.RandomState,
    ) -> Dict[str, np.ndarray]:
        cluster_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, n_clusters),
        )
        cluster_component = (contexts @ cluster_w)[:, clusters]

        feature_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, observed_features.shape[1]),
        )
        linear = contexts @ feature_w @ observed_features.T
        nonlinear = np.tanh(linear) + 0.25 * np.square(linear)
        feature_component = (
            (1.0 - self.feature_nonlinearity) * linear
            + self.feature_nonlinearity * nonlinear
        )

        identity_w = rng.normal(
            scale=1 / np.sqrt(self.dim_context),
            size=(self.dim_context, self.n_actions),
        )
        identity_component = contexts @ identity_w

        if hidden_embed.shape[1]:
            hidden_features = _one_hot_encoder().fit_transform(hidden_embed)
            hidden_w = rng.normal(
                scale=1 / np.sqrt(self.dim_context),
                size=(self.dim_context, hidden_features.shape[1]),
            )
            hidden_component = hidden_contexts @ hidden_w @ hidden_features.T
        else:
            hidden_action_w = rng.normal(size=self.n_actions)
            hidden_component = hidden_contexts[:, [0]] * hidden_action_w

        return {
            "cluster": _standardize_component(cluster_component),
            "feature": _standardize_component(feature_component),
            "identity": _standardize_component(identity_component),
            "hidden": _standardize_component(hidden_component),
        }
