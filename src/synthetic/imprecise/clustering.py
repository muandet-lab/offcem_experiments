"""Imprecise clustering methods and common uncertainty diagnostics."""
from dataclasses import dataclass
from time import time
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from clustering import compute_clusters


EPS = 1e-12


@dataclass
class ImpreciseClusteringResult:
    method: str
    labels: np.ndarray
    scores: Optional[np.ndarray]
    uncertainty: Dict[str, float]
    metadata: Dict
    candidates: Optional[List[np.ndarray]] = None


def fit_imprecise_clustering(
    action_features: np.ndarray,
    n_clusters: int,
    method: str,
    random_state: int,
    fuzzifier: float = 2.0,
    max_iter: int = 150,
    tolerance: float = 1e-5,
    rough_ratio: float = 1.25,
) -> ImpreciseClusteringResult:
    started = time()
    features = np.asarray(action_features, dtype=float)
    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        ).fit(features)
        labels = model.labels_.astype(int)
        scores = _inverse_distance_scores(
            features, model.cluster_centers_, power=2.0
        )
        result = ImpreciseClusteringResult(
            method=method,
            labels=labels,
            scores=scores,
            uncertainty=_probability_uncertainty(scores),
            metadata={"converged": True, "iterations": int(model.n_iter_)},
        )
    elif method == "fcm":
        result = _fit_fcm(
            features,
            n_clusters,
            random_state,
            fuzzifier,
            max_iter,
            tolerance,
        )
    elif method == "pcm":
        result = _fit_pcm(
            features,
            n_clusters,
            random_state,
            fuzzifier,
            max_iter,
            tolerance,
        )
    elif method == "rough":
        result = _fit_rough_kmeans(
            features,
            n_clusters,
            random_state,
            max_iter,
            tolerance,
            rough_ratio,
        )
    elif method == "ecm":
        result = _fit_ecm_optional(
            features,
            n_clusters,
            random_state,
        )
    else:
        raise ValueError(f"Unknown imprecise method: {method}")
    result.metadata["runtime_seconds"] = time() - started
    return result


def fixed_partition_result(
    action_features: np.ndarray,
    n_clusters: int,
    method: str,
    random_state: int,
    matched_labels: Optional[np.ndarray] = None,
    temperature: float = 10.0,
) -> ImpreciseClusteringResult:
    if method == "matched":
        if matched_labels is None:
            raise ValueError("matched requires matched_labels")
        labels = np.asarray(matched_labels, dtype=int).copy()
    else:
        labels = compute_clusters(
            action_features=action_features,
            n_clusters=n_clusters,
            method=method,
            balance="natural",
            random_state=random_state,
            temperature=temperature,
        )
    return ImpreciseClusteringResult(
        method=method,
        labels=labels,
        scores=None,
        uncertainty={},
        metadata={"converged": True, "runtime_seconds": 0.0},
    )


def sample_partitions(
    result: ImpreciseClusteringResult,
    n_samples: int,
    random_state: int,
    pcm_reject_threshold: float = 0.0,
) -> List[np.ndarray]:
    if n_samples < 1:
        return []
    rng = check_random_state(random_state)
    if result.method == "rough":
        return [
            _sample_candidates(result, rng)
            for _ in range(n_samples)
        ]
    if result.scores is None:
        return [result.labels.copy() for _ in range(n_samples)]

    probabilities = _row_normalize(result.scores)
    partitions = []
    for _ in range(n_samples):
        labels = np.array(
            [
                rng.choice(probabilities.shape[1], p=row)
                for row in probabilities
            ],
            dtype=int,
        )
        if result.method == "pcm" and pcm_reject_threshold > 0:
            rejected = result.scores.max(axis=1) < pcm_reject_threshold
            labels[rejected] = result.labels[rejected]
        partitions.append(
            _repair_empty_clusters(
                labels=labels,
                probabilities=probabilities,
                n_clusters=probabilities.shape[1],
            )
        )
    return partitions


def _fit_fcm(
    features,
    n_clusters,
    random_state,
    fuzzifier,
    max_iter,
    tolerance,
):
    if fuzzifier <= 1:
        raise ValueError("fuzzifier must be greater than 1")
    rng = check_random_state(random_state)
    membership = rng.dirichlet(
        np.ones(n_clusters), size=features.shape[0]
    )
    converged = False
    centers = None
    for iteration in range(1, max_iter + 1):
        powered = membership**fuzzifier
        centers = powered.T @ features
        centers /= np.maximum(powered.sum(axis=0)[:, None], EPS)
        distance_sq = _squared_distances(features, centers)
        updated = _fcm_membership(distance_sq, fuzzifier)
        if np.max(np.abs(updated - membership)) <= tolerance:
            membership = updated
            converged = True
            break
        membership = updated
    labels = _repair_empty_clusters(
        membership.argmax(axis=1).astype(int),
        membership,
        n_clusters,
    )
    return ImpreciseClusteringResult(
        method="fcm",
        labels=labels,
        scores=membership,
        uncertainty=_probability_uncertainty(membership),
        metadata={
            "converged": converged,
            "iterations": iteration,
            "fuzzifier": fuzzifier,
            "centers": centers.tolist(),
        },
    )


def _fit_pcm(
    features,
    n_clusters,
    random_state,
    fuzzifier,
    max_iter,
    tolerance,
):
    initial = _fit_fcm(
        features,
        n_clusters,
        random_state,
        fuzzifier,
        max_iter,
        tolerance,
    )
    centers = np.asarray(initial.metadata["centers"])
    distance_sq = _squared_distances(features, centers)
    membership = np.maximum(initial.scores, EPS)
    eta = (
        (membership**fuzzifier * distance_sq).sum(axis=0)
        / np.maximum((membership**fuzzifier).sum(axis=0), EPS)
    )
    eta = np.maximum(eta, EPS)

    converged = False
    typicality = membership.copy()
    for iteration in range(1, max_iter + 1):
        distance_sq = _squared_distances(features, centers)
        updated = 1.0 / (
            1.0
            + (distance_sq / eta[None, :])
            ** (1.0 / (fuzzifier - 1.0))
        )
        powered = updated**fuzzifier
        new_centers = powered.T @ features
        new_centers /= np.maximum(powered.sum(axis=0)[:, None], EPS)
        if np.max(np.abs(new_centers - centers)) <= tolerance:
            centers = new_centers
            typicality = updated
            converged = True
            break
        centers = new_centers
        typicality = updated

    labels = _repair_empty_clusters(
        typicality.argmax(axis=1).astype(int),
        _row_normalize(typicality),
        n_clusters,
    )
    uncertainty = _typicality_uncertainty(typicality)
    uncertainty["coincident_center_pairs"] = _coincident_center_pairs(centers)
    return ImpreciseClusteringResult(
        method="pcm",
        labels=labels,
        scores=typicality,
        uncertainty=uncertainty,
        metadata={
            "converged": converged,
            "iterations": iteration,
            "fuzzifier": fuzzifier,
            "eta": eta.tolist(),
            "centers": centers.tolist(),
        },
    )


def _fit_rough_kmeans(
    features,
    n_clusters,
    random_state,
    max_iter,
    tolerance,
    rough_ratio,
):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit(features)
    centers = model.cluster_centers_
    candidates = None
    converged = False
    for iteration in range(1, max_iter + 1):
        distances = np.sqrt(_squared_distances(features, centers))
        nearest = distances.argmin(axis=1)
        candidates = []
        for index, first in enumerate(nearest):
            threshold = max(distances[index, first] * rough_ratio, EPS)
            plausible = np.flatnonzero(distances[index] <= threshold)
            if plausible.size == 0:
                plausible = np.array([first])
            candidates.append(plausible.astype(int))

        new_centers = centers.copy()
        for cluster in range(n_clusters):
            lower = np.array(
                [
                    index
                    for index, plausible in enumerate(candidates)
                    if plausible.size == 1 and plausible[0] == cluster
                ],
                dtype=int,
            )
            boundary = np.array(
                [
                    index
                    for index, plausible in enumerate(candidates)
                    if plausible.size > 1 and cluster in plausible
                ],
                dtype=int,
            )
            if lower.size and boundary.size:
                new_centers[cluster] = (
                    0.7 * features[lower].mean(axis=0)
                    + 0.3 * features[boundary].mean(axis=0)
                )
            elif lower.size:
                new_centers[cluster] = features[lower].mean(axis=0)
            elif boundary.size:
                new_centers[cluster] = features[boundary].mean(axis=0)
        if np.max(np.abs(new_centers - centers)) <= tolerance:
            centers = new_centers
            converged = True
            break
        centers = new_centers

    distances = np.sqrt(_squared_distances(features, centers))
    labels = distances.argmin(axis=1).astype(int)
    candidate_counts = np.array([len(item) for item in candidates])
    uncertainty = {
        "boundary_fraction": float(np.mean(candidate_counts > 1)),
        "candidate_count_mean": float(candidate_counts.mean()),
        "candidate_count_max": int(candidate_counts.max()),
    }
    return ImpreciseClusteringResult(
        method="rough",
        labels=labels,
        scores=_inverse_distance_scores(features, centers, power=2.0),
        uncertainty=uncertainty,
        metadata={
            "converged": converged,
            "iterations": iteration,
            "rough_ratio": rough_ratio,
            "centers": centers.tolist(),
        },
        candidates=candidates,
    )


def _fit_ecm_optional(features, n_clusters, random_state):
    try:
        import evclust
    except ImportError as error:
        raise RuntimeError(
            "ECM requires the optional 'evclust' package. "
            "Install it in the offcem environment before using --methods ecm."
        ) from error
    raise RuntimeError(
        "The installed evclust API must be pinned and validated before ECM is "
        "enabled; the runner refuses to guess an incompatible API."
    )


def _fcm_membership(distance_sq, fuzzifier):
    membership = np.zeros_like(distance_sq)
    zero = distance_sq <= EPS
    zero_rows = zero.any(axis=1)
    for row in np.flatnonzero(zero_rows):
        membership[row, zero[row]] = 1.0 / zero[row].sum()
    nonzero_rows = ~zero_rows
    inverse = np.maximum(
        distance_sq[nonzero_rows], EPS
    ) ** (-1.0 / (fuzzifier - 1.0))
    membership[nonzero_rows] = inverse / inverse.sum(axis=1, keepdims=True)
    return membership


def _probability_uncertainty(scores):
    probabilities = _row_normalize(scores)
    ordered = np.sort(probabilities, axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, EPS)), axis=1
    )
    normalized_entropy = entropy / max(np.log(probabilities.shape[1]), EPS)
    margin = ordered[:, -1] - ordered[:, -2]
    return {
        "entropy_mean": float(normalized_entropy.mean()),
        "entropy_q90": float(np.quantile(normalized_entropy, 0.90)),
        "margin_mean": float(margin.mean()),
        "margin_q10": float(np.quantile(margin, 0.10)),
        "ambiguous_fraction_margin_lt_0.1": float(np.mean(margin < 0.1)),
    }


def _typicality_uncertainty(scores):
    ordered = np.sort(scores, axis=1)
    maximum = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    return {
        "max_typicality_mean": float(maximum.mean()),
        "max_typicality_q10": float(np.quantile(maximum, 0.10)),
        "total_typicality_mean": float(scores.sum(axis=1).mean()),
        "typicality_margin_mean": float(margin.mean()),
        "low_typicality_fraction_lt_0.5": float(np.mean(maximum < 0.5)),
    }


def _sample_candidates(result, rng):
    labels = result.labels.copy()
    for index, plausible in enumerate(result.candidates):
        labels[index] = rng.choice(plausible)
    probabilities = (
        _row_normalize(result.scores)
        if result.scores is not None
        else None
    )
    return _repair_empty_clusters(
        labels,
        probabilities,
        (
            probabilities.shape[1]
            if probabilities is not None
            else int(labels.max()) + 1
        ),
    )


def _repair_empty_clusters(labels, probabilities, n_clusters):
    labels = np.asarray(labels, dtype=int).copy()
    counts = np.bincount(labels, minlength=n_clusters)
    for cluster in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        donor_actions = np.flatnonzero(labels == donor)
        if donor_actions.size <= 1:
            continue
        if probabilities is None:
            action = donor_actions[0]
        else:
            action = donor_actions[
                np.argmax(probabilities[donor_actions, cluster])
            ]
        labels[action] = cluster
        counts[donor] -= 1
        counts[cluster] += 1
    return labels


def _inverse_distance_scores(features, centers, power):
    distance_sq = _squared_distances(features, centers)
    inverse = np.maximum(distance_sq, EPS) ** (-power / 2.0)
    return _row_normalize(inverse)


def _squared_distances(features, centers):
    return np.sum(
        (features[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )


def _row_normalize(scores):
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    totals = scores.sum(axis=1, keepdims=True)
    zero = totals[:, 0] <= EPS
    if zero.any():
        scores[zero] = 1.0
        totals = scores.sum(axis=1, keepdims=True)
    return scores / totals


def _coincident_center_pairs(centers, tolerance=1e-4):
    count = 0
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            if np.linalg.norm(centers[first] - centers[second]) <= tolerance:
                count += 1
    return count
