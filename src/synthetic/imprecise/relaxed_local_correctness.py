"""Population calculations for approximate local correctness."""
from typing import Dict
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def relaxed_local_correctness_bound(
    expected_reward: np.ndarray,
    prediction: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    clusters: np.ndarray,
) -> Dict[str, float]:
    """Evaluate the exact population OffCEM bias and its epsilon-TV bound.

    The population is the empirical collection of contexts represented by the
    rows of the input arrays. Epsilon is computed separately for every
    context-cluster pair as the oscillation of q - q_hat.
    """
    expected_reward = np.asarray(expected_reward, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    pi_b = np.asarray(pi_b, dtype=float)
    pi_e = np.asarray(pi_e, dtype=float)
    clusters = np.asarray(clusters, dtype=int)
    if expected_reward.shape != prediction.shape:
        raise ValueError("expected_reward and prediction must have equal shape")
    if expected_reward.shape != pi_b.shape or pi_b.shape != pi_e.shape:
        raise ValueError("reward and policy arrays must have equal shape")
    if expected_reward.shape[1] != clusters.size:
        raise ValueError("clusters must contain one label per action")

    residual = expected_reward - prediction
    direct = np.sum(pi_e * prediction, axis=1)
    correction = np.zeros(expected_reward.shape[0])
    bound = np.zeros(expected_reward.shape[0])
    weighted_epsilon = np.zeros(expected_reward.shape[0])
    weighted_tv = np.zeros(expected_reward.shape[0])

    for cluster in np.unique(clusters):
        mask = clusters == cluster
        pi_b_cluster = pi_b[:, mask].sum(axis=1)
        pi_e_cluster = pi_e[:, mask].sum(axis=1)
        pi_b_conditional = pi_b[:, mask] / np.maximum(
            pi_b_cluster[:, None], EPS
        )
        pi_e_conditional = pi_e[:, mask] / np.maximum(
            pi_e_cluster[:, None], EPS
        )
        epsilon = np.ptp(residual[:, mask], axis=1)
        total_variation = 0.5 * np.sum(
            np.abs(pi_b_conditional - pi_e_conditional), axis=1
        )
        correction += pi_e_cluster * np.sum(
            pi_b_conditional * residual[:, mask], axis=1
        )
        bound += pi_e_cluster * epsilon * total_variation
        weighted_epsilon += pi_e_cluster * epsilon
        weighted_tv += pi_e_cluster * total_variation

    population_value = float(np.mean(direct + correction))
    true_value = float(np.mean(np.sum(pi_e * expected_reward, axis=1)))
    bias = population_value - true_value
    mean_bound = float(np.mean(bound))
    required_fraction = (
        abs(bias) / mean_bound
        if mean_bound > EPS
        else (0.0 if abs(bias) <= EPS else np.inf)
    )
    return {
        "population_offcem_value": population_value,
        "population_true_value": true_value,
        "population_bias": float(bias),
        "absolute_population_bias": float(abs(bias)),
        "epsilon_tv_bound": mean_bound,
        "bound_covers_population_bias": bool(abs(bias) <= mean_bound + 1e-10),
        "required_epsilon_fraction": float(required_fraction),
        "target_weighted_epsilon": float(np.mean(weighted_epsilon)),
        "target_weighted_tv": float(np.mean(weighted_tv)),
    }


def summarize_relaxed_records(
    records: Iterable[Dict],
    output_dir,
) -> Dict:
    """Write partition-level and aggregate summaries."""
    partition_rows = []
    robust_rows = []
    fit_rows = []
    for record in records:
        if "error" in record:
            continue
        fit_rows.append(
            {
                "ambiguity_fraction": record["ambiguity_fraction"],
                "rough_ratio": record["rough_ratio"],
                "seed": record["seed"],
                **record["rough_uncertainty"],
            }
        )
        valid = [
            partition
            for partition in record["partitions"]
            if "error" not in partition
        ]
        for partition in valid:
            bound = partition["relaxed_bound"]["epsilon_tv_bound"]
            population_value = partition["relaxed_bound"][
                "population_offcem_value"
            ]
            true_value = partition["relaxed_bound"]["population_true_value"]
            estimate = partition["estimate"]
            row = {
                "ambiguity_fraction": record["ambiguity_fraction"],
                "rough_ratio": record["rough_ratio"],
                "seed": record["seed"],
                "source": partition["source"],
                "draw": partition["draw"],
                "estimate": estimate,
                "population_offcem_value": population_value,
                "population_true_value": true_value,
                "population_bias": partition["relaxed_bound"][
                    "population_bias"
                ],
                "epsilon_tv_bound": bound,
                "population_interval_width": 2.0 * bound,
                "population_coverage": partition["relaxed_bound"][
                    "bound_covers_population_bias"
                ],
                "sample_interval_coverage": bool(
                    estimate - bound <= true_value <= estimate + bound
                ),
                "required_epsilon_fraction": partition["relaxed_bound"][
                    "required_epsilon_fraction"
                ],
                "target_weighted_epsilon": partition["relaxed_bound"][
                    "target_weighted_epsilon"
                ],
                "target_weighted_tv": partition["relaxed_bound"][
                    "target_weighted_tv"
                ],
                "dm_2s": partition["diagnostics"]["dm_2s"],
                "dm_policy_weighted": partition["diagnostics"][
                    "dm_policy_weighted"
                ],
                "ari_primary": partition["ari_primary"],
            }
            for fraction in (0.25, 0.5, 0.75, 1.0):
                row[f"coverage_epsilon_{fraction:g}"] = bool(
                    abs(row["population_bias"])
                    <= fraction * bound + 1e-10
                )
            partition_rows.append(row)

        admissible = [
            partition
            for partition in valid
            if partition["source"] in {"rough_hard", "rough_sample"}
        ]
        if admissible:
            lower = min(
                item["relaxed_bound"]["population_offcem_value"]
                - item["relaxed_bound"]["epsilon_tv_bound"]
                for item in admissible
            )
            upper = max(
                item["relaxed_bound"]["population_offcem_value"]
                + item["relaxed_bound"]["epsilon_tv_bound"]
                for item in admissible
            )
            true_value = admissible[0]["relaxed_bound"][
                "population_true_value"
            ]
            robust_rows.append(
                {
                    "ambiguity_fraction": record["ambiguity_fraction"],
                    "rough_ratio": record["rough_ratio"],
                    "seed": record["seed"],
                    "n_partitions": len(admissible),
                    "robust_lower": lower,
                    "robust_upper": upper,
                    "robust_width": upper - lower,
                    "robust_coverage": bool(
                        lower <= true_value <= upper
                    ),
                }
            )

    partitions = pd.DataFrame(partition_rows)
    fits = pd.DataFrame(fit_rows)
    robust = pd.DataFrame(robust_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    partitions.to_csv(output_dir / "partition_results.csv", index=False)
    fits.to_csv(output_dir / "rough_fit_summary.csv", index=False)
    robust.to_csv(output_dir / "robust_intervals.csv", index=False)

    aggregate = pd.DataFrame()
    if not partitions.empty:
        metrics = [
            "epsilon_tv_bound",
            "population_interval_width",
            "population_coverage",
            "sample_interval_coverage",
            "required_epsilon_fraction",
            "target_weighted_epsilon",
            "target_weighted_tv",
            "dm_2s",
            "dm_policy_weighted",
            "ari_primary",
            "coverage_epsilon_0.25",
            "coverage_epsilon_0.5",
            "coverage_epsilon_0.75",
            "coverage_epsilon_1",
        ]
        aggregate = (
            partitions.groupby(
                ["ambiguity_fraction", "rough_ratio", "source"]
            )[metrics]
            .mean()
            .reset_index()
        )
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    return {
        "n_fits": len(fits),
        "n_partitions": len(partitions),
        "aggregate": aggregate.to_dict(orient="records"),
        "robust": robust.to_dict(orient="records"),
    }
