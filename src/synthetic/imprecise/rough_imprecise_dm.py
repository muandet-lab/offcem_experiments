"""Rough-set direct-method bounds under context-action support deficiency."""
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np
import pandas as pd


EPS = 1e-12


def rough_dm_bounds(
    prediction: np.ndarray,
    target_policy: np.ndarray,
    support_mask: np.ndarray,
    action_features: np.ndarray,
    hard_labels: np.ndarray,
    candidates: List[np.ndarray],
    factual_action: np.ndarray,
    factual_reward: np.ndarray,
    gamma: float,
    outcome_lower: float,
    outcome_upper: float,
    calibration_quantile: float = 0.9,
    n_neighbors: int = 5,
    min_calibration_count: int = 20,
    expected_reward: np.ndarray = None,
) -> Dict:
    """Construct rough-DM bounds for actions outside behavior-policy support.

    Rough lower-approximation members are the preferred supported analogues.
    Boundary members are used only when a candidate cluster has no supported
    lower members, followed by all supported actions as a final fallback.

    For each candidate cluster, the interval is the envelope of the nearest
    analogue predictions, enlarged by a residual calibration radius and
    ``gamma`` times normalized action-feature distance. Candidate-cluster
    intervals are then unioned. This is an explicit extrapolation assumption,
    not an implication of rough K-means alone.
    """
    prediction = np.asarray(prediction, dtype=float)
    target_policy = np.asarray(target_policy, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    action_features = np.asarray(action_features, dtype=float)
    hard_labels = np.asarray(hard_labels, dtype=int)
    factual_action = np.asarray(factual_action, dtype=int)
    factual_reward = np.asarray(factual_reward, dtype=float)
    support_matrix = _support_matrix(
        support_mask,
        n_contexts=prediction.shape[0],
        n_actions=prediction.shape[1],
    )
    _validate_inputs(
        prediction=prediction,
        target_policy=target_policy,
        support_matrix=support_matrix,
        action_features=action_features,
        hard_labels=hard_labels,
        candidates=candidates,
        factual_action=factual_action,
        factual_reward=factual_reward,
        gamma=gamma,
        outcome_lower=outcome_lower,
        outcome_upper=outcome_upper,
        calibration_quantile=calibration_quantile,
        n_neighbors=n_neighbors,
        min_calibration_count=min_calibration_count,
    )

    unsupported = ~support_matrix
    clipped_prediction = np.clip(
        prediction, outcome_lower, outcome_upper
    )
    lower_prediction = clipped_prediction.copy()
    upper_prediction = clipped_prediction.copy()
    global_radius, cluster_radii = _calibration_radii(
        prediction=prediction,
        hard_labels=hard_labels,
        factual_action=factual_action,
        factual_reward=factual_reward,
        quantile=calibration_quantile,
        min_count=min_calibration_count,
    )
    lower_members, upper_members = _rough_approximations(
        candidates=candidates,
        n_clusters=int(hard_labels.max()) + 1,
    )
    distance_scale = _distance_scale(
        action_features=action_features,
        support_matrix=support_matrix,
    )

    lower_fallbacks = 0
    upper_fallbacks = 0
    global_fallbacks = 0
    candidate_evaluations = 0
    selected_analogue_counts = []
    for context in range(prediction.shape[0]):
        supported_actions = np.flatnonzero(support_matrix[context])
        for action in np.flatnonzero(unsupported[context]):
            action_lowers = []
            action_uppers = []
            plausible = np.asarray(candidates[action], dtype=int)
            for cluster in plausible:
                candidate_evaluations += 1
                pool = lower_members[cluster]
                pool = pool[support_matrix[context, pool]]
                if pool.size == 0:
                    lower_fallbacks += 1
                    pool = upper_members[cluster]
                    pool = pool[support_matrix[context, pool]]
                if pool.size == 0:
                    upper_fallbacks += 1
                    global_fallbacks += 1
                    pool = supported_actions
                selected, distances = _nearest_analogues(
                    action=action,
                    pool=pool,
                    action_features=action_features,
                    n_neighbors=n_neighbors,
                    distance_scale=distance_scale,
                )
                selected_analogue_counts.append(selected.size)
                radius = cluster_radii.get(int(cluster), global_radius)
                analogue_prediction = clipped_prediction[context, selected]
                expansion = radius + gamma * distances
                action_lowers.append(
                    np.min(analogue_prediction - expansion)
                )
                action_uppers.append(
                    np.max(analogue_prediction + expansion)
                )
            lower_prediction[context, action] = np.clip(
                np.min(action_lowers),
                outcome_lower,
                outcome_upper,
            )
            upper_prediction[context, action] = np.clip(
                np.max(action_uppers),
                outcome_lower,
                outcome_upper,
            )

    supported_value = float(
        np.mean(
            np.sum(target_policy * clipped_prediction * support_matrix, axis=1)
        )
    )
    lower_value = float(np.mean(np.sum(target_policy * lower_prediction, axis=1)))
    upper_value = float(np.mean(np.sum(target_policy * upper_prediction, axis=1)))
    unsupported_target_mass = float(
        np.mean(np.sum(target_policy * unsupported, axis=1))
    )
    manski_lower = supported_value + unsupported_target_mass * outcome_lower
    manski_upper = supported_value + unsupported_target_mass * outcome_upper
    result = {
        "lower_value": lower_value,
        "upper_value": upper_value,
        "midpoint_value": 0.5 * (lower_value + upper_value),
        "interval_width": upper_value - lower_value,
        "supported_dm_value": supported_value,
        "unsupported_target_mass": unsupported_target_mass,
        "manski_lower": float(manski_lower),
        "manski_upper": float(manski_upper),
        "manski_width": float(manski_upper - manski_lower),
        "width_fraction_of_manski": float(
            (upper_value - lower_value)
            / max(manski_upper - manski_lower, EPS)
        ),
        "global_calibration_radius": global_radius,
        "cluster_calibration_radius_mean": float(
            np.mean(list(cluster_radii.values()))
        ),
        "distance_scale": distance_scale,
        "candidate_evaluations": candidate_evaluations,
        "lower_core_fallback_fraction": float(
            lower_fallbacks / max(candidate_evaluations, 1)
        ),
        "upper_approximation_fallback_fraction": float(
            upper_fallbacks / max(candidate_evaluations, 1)
        ),
        "global_fallback_fraction": float(
            global_fallbacks / max(candidate_evaluations, 1)
        ),
        "analogue_count_mean": float(
            np.mean(selected_analogue_counts)
            if selected_analogue_counts
            else 0.0
        ),
        "lower_prediction": lower_prediction,
        "upper_prediction": upper_prediction,
    }
    if expected_reward is not None:
        expected_reward = np.asarray(expected_reward, dtype=float)
        if expected_reward.shape != prediction.shape:
            raise ValueError(
                "expected_reward must have the same shape as prediction"
            )
        covered = (
            (lower_prediction <= expected_reward + 1e-10)
            & (expected_reward <= upper_prediction + 1e-10)
        )
        unsupported_weight = target_policy * unsupported
        if unsupported.any():
            pair_coverage = float(covered[unsupported].mean())
            weighted_coverage = float(
                np.sum(unsupported_weight * covered)
                / max(unsupported_weight.sum(), EPS)
            )
        else:
            pair_coverage = 1.0
            weighted_coverage = 1.0
        result.update(
            {
                "unsupported_pair_coverage": pair_coverage,
                "unsupported_policy_weighted_coverage": weighted_coverage,
            }
        )
    return result


def summarize_rough_dm_records(
    records: Iterable[Dict],
    output_dir: Path,
) -> Dict:
    """Write per-bound and aggregate summaries."""
    rows = []
    fit_rows = []
    for record in records:
        if "error" in record:
            continue
        fit_rows.append(
            {
                "deficient_action_fraction": record[
                    "deficient_action_fraction"
                ],
                "support_mode": record["support_mode"],
                "rough_ratio": record["rough_ratio"],
                "seed": record["seed"],
                "true_policy_value": record["true_policy_value"],
                "sample_true_policy_value": record[
                    "sample_true_policy_value"
                ],
                "standard_dm_value": record["standard_dm_value"],
                "standard_dm_sample_error": record[
                    "standard_dm_sample_error"
                ],
                "outcome_range_covers_population": record[
                    "outcome_range_covers_population"
                ],
                **record["rough_uncertainty"],
            }
        )
        for bound in record["bounds"]:
            if "error" in bound:
                continue
            rows.append(
                {
                    "deficient_action_fraction": record[
                        "deficient_action_fraction"
                    ],
                    "support_mode": record["support_mode"],
                    "rough_ratio": record["rough_ratio"],
                    "seed": record["seed"],
                    **bound,
                }
            )

    bounds = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(output_dir / "bound_results.csv", index=False)
    fits.to_csv(output_dir / "rough_fit_summary.csv", index=False)

    aggregate = pd.DataFrame()
    if not bounds.empty:
        metrics = [
            "lower_value",
            "upper_value",
            "interval_width",
            "width_fraction_of_manski",
            "conditional_coverage",
            "population_coverage",
            "manski_conditional_coverage",
            "manski_population_coverage",
            "unsupported_pair_coverage",
            "unsupported_policy_weighted_coverage",
            "lower_core_fallback_fraction",
            "upper_approximation_fallback_fraction",
            "global_fallback_fraction",
        ]
        aggregate = (
            bounds.groupby(
                [
                    "support_mode",
                    "deficient_action_fraction",
                    "rough_ratio",
                    "gamma",
                ]
            )[metrics]
            .mean()
            .reset_index()
        )
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    return {
        "n_fits": len(fits),
        "n_bounds": len(bounds),
        "aggregate": aggregate.to_dict(orient="records"),
    }


def _validate_inputs(
    prediction,
    target_policy,
    support_matrix,
    action_features,
    hard_labels,
    candidates,
    factual_action,
    factual_reward,
    gamma,
    outcome_lower,
    outcome_upper,
    calibration_quantile,
    n_neighbors,
    min_calibration_count,
):
    if prediction.ndim != 2 or prediction.shape != target_policy.shape:
        raise ValueError("prediction and target_policy must be equal 2D arrays")
    n_contexts, n_actions = prediction.shape
    if support_matrix.shape != (n_contexts, n_actions):
        raise ValueError("support must match contexts and actions")
    if np.any(~support_matrix.any(axis=1)):
        raise ValueError("each context must support at least one action")
    if action_features.shape[0] != n_actions:
        raise ValueError("action_features must contain one row per action")
    if hard_labels.shape != (n_actions,) or len(candidates) != n_actions:
        raise ValueError("rough assignments must contain one entry per action")
    if factual_action.shape != (n_contexts,) or factual_reward.shape != (
        n_contexts,
    ):
        raise ValueError("factual arrays must contain one value per context")
    if np.any(
        ~support_matrix[np.arange(n_contexts), factual_action]
    ):
        raise ValueError("factual actions must lie inside behavior support")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if not outcome_lower < outcome_upper:
        raise ValueError("outcome_lower must be smaller than outcome_upper")
    if not 0 <= calibration_quantile <= 1:
        raise ValueError("calibration_quantile must be in [0, 1]")
    if n_neighbors < 1 or min_calibration_count < 1:
        raise ValueError("neighbor and calibration counts must be positive")


def _calibration_radii(
    prediction,
    hard_labels,
    factual_action,
    factual_reward,
    quantile,
    min_count,
):
    factual_prediction = prediction[
        np.arange(factual_action.size), factual_action
    ]
    residual = np.abs(factual_reward - factual_prediction)
    global_radius = float(np.quantile(residual, quantile))
    cluster_radii = {}
    observed_clusters = hard_labels[factual_action]
    for cluster in np.unique(hard_labels):
        selected = residual[observed_clusters == cluster]
        cluster_radii[int(cluster)] = float(
            np.quantile(selected, quantile)
            if selected.size >= min_count
            else global_radius
        )
    return global_radius, cluster_radii


def _rough_approximations(candidates, n_clusters):
    lower_members = []
    upper_members = []
    for cluster in range(n_clusters):
        lower_members.append(
            np.array(
                [
                    action
                    for action, plausible in enumerate(candidates)
                    if len(plausible) == 1
                    and int(plausible[0]) == cluster
                ],
                dtype=int,
            )
        )
        upper_members.append(
            np.array(
                [
                    action
                    for action, plausible in enumerate(candidates)
                    if cluster in plausible
                ],
                dtype=int,
            )
        )
    return lower_members, upper_members


def _distance_scale(action_features, support_matrix):
    nearest_distances = []
    for support in np.unique(support_matrix, axis=0):
        unsupported = np.flatnonzero(~support)
        supported = np.flatnonzero(support)
        if unsupported.size == 0:
            continue
        distances = np.sqrt(
            np.sum(
                (
                    action_features[unsupported, np.newaxis, :]
                    - action_features[supported, :]
                )
                ** 2,
                axis=2,
            )
        )
        nearest_distances.extend(distances.min(axis=1).tolist())
    positive = np.asarray(nearest_distances)
    positive = positive[positive > EPS]
    return float(np.median(positive)) if positive.size else 1.0


def _support_matrix(support, n_contexts, n_actions):
    if support.shape == (n_actions,):
        return np.broadcast_to(
            support[np.newaxis, :], (n_contexts, n_actions)
        )
    if support.shape == (n_contexts, n_actions):
        return support
    raise ValueError(
        "support_mask must be action-level or context-action-level"
    )


def _nearest_analogues(
    action,
    pool,
    action_features,
    n_neighbors,
    distance_scale,
):
    distances = np.sqrt(
        np.sum(
            (action_features[pool] - action_features[action]) ** 2,
            axis=1,
        )
    )
    order = np.argsort(distances)[: min(n_neighbors, pool.size)]
    return pool[order], distances[order] / max(distance_scale, EPS)
