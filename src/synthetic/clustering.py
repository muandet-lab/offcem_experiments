"""Clustering algorithms for action space partitioning.

Each clustering function has the signature:
    fn(action_features, n_clusters, random_state) -> 1D int array of shape (n_actions,)
"""
import enum

import numpy as np
from obp.dataset import linear_behavior_policy
from obp.utils import sample_action_fast
from obp.utils import softmax
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.utils import check_random_state


class ClusterMethod(enum.Enum):
    ORIGINAL = "original"
    KMEANS = "kmeans"
    SPECTRAL = "spectral"
    AGGLOMERATIVE = "agglomerative"
    RANDOM = "random"
    SHUFFLED_PARTITION = "shuffled_partition"
    FEATURE_BUCKET = "feature_bucket"


class BalanceType(enum.Enum):
    NATURAL = "natural"
    BALANCED = "balanced"
    POWER_LAW = "power_law"


def cluster_original(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
    temperature: float = 10.0,
) -> np.ndarray:
    """The paper's original method: linear_behavior_policy -> softmax -> sample -> rankdata.

    Lower temperature -> sharper softmax -> more feature-coherent clusters.
    Paper default is 10.0 (flat). Use 1.0 for near-deterministic assignment.
    """
    cluster_logits = linear_behavior_policy(
        context=action_features,
        action_context=np.eye(n_clusters),
        random_state=random_state,
    )
    clusters = sample_action_fast(
        softmax(cluster_logits / temperature), random_state=random_state
    )
    clusters = rankdata(-clusters, method="dense") - 1
    return clusters


def cluster_kmeans(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(action_features)


def cluster_spectral(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    n_neighbors = min(20, action_features.shape[0] - 1)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state,
        assign_labels="kmeans",
    )
    return sc.fit_predict(action_features)


def cluster_agglomerative(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return agg.fit_predict(action_features)


def cluster_feature_bucket(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    """Deterministic feature-derived partition for the feature-coherent benchmark.

    Projects actions onto their dominant feature axis (top principal component
    via SVD) and slices that ordering into ``n_clusters`` equal-frequency bins.

    Unlike ``original`` -- a stochastic softmax sample of linear logits, which is
    non-geometric (ARI~0 with kmeans) so no feature-based clusterer can recover
    it -- this partition is a deterministic function of the action features and
    is therefore recoverable by feature-based clustering. When ``dataset.py``
    builds its per-cluster reward on top of this partition, the reward genuinely
    tracks features, letting us test whether OffCEM's local correctness is
    achievable via a practical feature clustering (the question the default
    feature-arbitrary benchmark cannot answer). Equal-frequency bins also fix
    cluster sizes, removing the size-imbalance tail confound (finding D-3).

    ``random_state`` is accepted for signature compatibility but unused: SVD and
    equal-frequency bucketing are fully deterministic given the features. The
    sign of the principal component is arbitrary, but flipping it only reverses
    the bin labels, leaving the grouping (and thus ARI / dense relabeling) intact.
    """
    X = np.asarray(action_features, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[0]
    ranks = rankdata(proj, method="ordinal") - 1  # 0 .. n_actions-1
    n_actions = X.shape[0]
    clusters = (ranks * n_clusters) // n_actions
    return clusters.astype(int)


def cluster_random(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    rng = check_random_state(random_state)
    return rng.randint(0, n_clusters, size=action_features.shape[0])


def cluster_shuffled_partition(
    action_features: np.ndarray,
    n_clusters: int,
    random_state: int = 12345,
) -> np.ndarray:
    """Shuffle actions, then slice into equal-sized contiguous clusters."""
    rng = check_random_state(random_state)
    n_actions = action_features.shape[0]
    perm = rng.permutation(n_actions)
    clusters = np.empty(n_actions, dtype=int)
    base = n_actions // n_clusters
    remainder = n_actions % n_clusters
    idx = 0
    for c in range(n_clusters):
        size = base + (1 if c < remainder else 0)
        clusters[perm[idx : idx + size]] = c
        idx += size
    return clusters


def corrupt_partition(
    true_clusters: np.ndarray,
    p: float,
    random_state: int = 12345,
) -> np.ndarray:
    """Graded corruption of a known-good partition.

    Reassign a fraction ``p`` of actions to uniformly-random clusters.
    ``p=0`` returns the exact partition; ``p=1`` is fully random. The number of
    clusters is preserved. This builds a continuous optimal->random clustering-
    quality axis for OffCEM (the relMSE-vs-p crossover and the local-correctness
    violation-vs-p curve). Unlike the CLUSTER_FUNCTIONS, this needs the true
    partition rather than action features, so it is a standalone helper.
    """
    rng = check_random_state(random_state)
    clusters = np.asarray(true_clusters, dtype=int).copy()
    n = clusters.shape[0]
    n_clusters = int(clusters.max()) + 1
    n_corrupt = int(round(float(p) * n))
    if n_corrupt > 0:
        idx = rng.choice(n, size=n_corrupt, replace=False)
        clusters[idx] = rng.randint(0, n_clusters, size=n_corrupt)
    clusters = rankdata(clusters, method="dense") - 1
    return clusters.astype(int)


CLUSTER_FUNCTIONS = {
    ClusterMethod.ORIGINAL: cluster_original,
    ClusterMethod.KMEANS: cluster_kmeans,
    ClusterMethod.SPECTRAL: cluster_spectral,
    ClusterMethod.AGGLOMERATIVE: cluster_agglomerative,
    ClusterMethod.RANDOM: cluster_random,
    ClusterMethod.SHUFFLED_PARTITION: cluster_shuffled_partition,
    ClusterMethod.FEATURE_BUCKET: cluster_feature_bucket,
}


def compute_clusters(
    action_features: np.ndarray,
    n_clusters: int,
    method: str = "original",
    balance: str = "natural",
    random_state: int = 12345,
    temperature: float = 10.0,
) -> np.ndarray:
    """Main entry point: cluster + optional rebalance.
    Returns 1D array shape (n_actions,) with dense values in [0, actual_n_clusters).
    temperature only affects the 'original' method.
    """
    fn = CLUSTER_FUNCTIONS[ClusterMethod(method)]
    if method == "original":
        clusters = fn(action_features, n_clusters, random_state=random_state, temperature=temperature)
    else:
        clusters = fn(action_features, n_clusters, random_state=random_state)
    if balance != "natural":
        clusters = rebalance_clusters(clusters, n_clusters, balance, random_state)
    clusters = rankdata(clusters, method="dense") - 1
    return clusters.astype(int)


def rebalance_clusters(
    clusters: np.ndarray,
    n_clusters: int,
    balance: str,
    random_state: int = 12345,
) -> np.ndarray:
    """Redistribute actions across clusters to achieve desired balance.

    Preserves relative ordering within each original cluster so nearby
    actions in feature space stay together as much as possible.
    """
    rng = check_random_state(random_state)
    n_actions = len(clusters)
    target_sizes = _generate_target_sizes(n_actions, n_clusters, balance, rng)

    order = np.argsort(clusters, kind="stable")
    new_clusters = np.empty(n_actions, dtype=int)
    idx = 0
    for c, size in enumerate(target_sizes):
        new_clusters[order[idx : idx + size]] = c
        idx += size
    return new_clusters


def _generate_target_sizes(
    n_actions: int,
    n_clusters: int,
    balance: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    balance_type = BalanceType(balance)
    if balance_type == BalanceType.BALANCED:
        base = n_actions // n_clusters
        remainder = n_actions % n_clusters
        sizes = np.full(n_clusters, base)
        sizes[:remainder] += 1
    elif balance_type == BalanceType.POWER_LAW:
        alpha = 1.0
        weights = 1.0 / np.arange(1, n_clusters + 1) ** alpha
        weights /= weights.sum()
        sizes = np.maximum(1, np.round(weights * n_actions).astype(int))
        diff = n_actions - sizes.sum()
        indices = rng.permutation(n_clusters)
        for i in range(abs(diff)):
            sizes[indices[i % n_clusters]] += int(np.sign(diff))
    else:
        raise ValueError(f"Unknown balance type: {balance}")
    return sizes


def clusters_to_onehot_3d(
    clusters_1d: np.ndarray,
    n_users: int,
) -> np.ndarray:
    """Convert 1D cluster assignments to the 3D one-hot format used by ope.py.
    Returns shape (n_users, n_actions, n_clusters).
    """
    n_actions = len(clusters_1d)
    n_clusters = int(clusters_1d.max()) + 1
    mat = np.zeros((n_actions, n_clusters))
    mat[np.arange(n_actions), clusters_1d] = 1
    return np.tile(mat, (n_users, 1, 1))
