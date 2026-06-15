"""Shared experiment utilities for membership-uncertainty studies."""
import json
import os
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import torch

from applicability.dataset import ApplicabilityDataset
from applicability.diagnostics import reward_diagnostics
from applicability.estimators import estimate_partition_methods
from applicability.estimators import estimate_shared_baselines
from applicability.estimators import fit_action_reward_model
from applicability.estimators import fit_two_stage_model
from clustering import clusters_to_onehot_3d
from imprecise.clustering import fit_imprecise_clustering
from imprecise.clustering import fixed_partition_result


IMPRECISE_METHODS = {"fcm", "pcm", "rough", "ecm"}


def parse_alpha(value: str) -> List[float]:
    alpha = [float(item) for item in value.split(",")]
    if len(alpha) != 4:
        raise ValueError("--reward-alpha requires four comma-separated values")
    if any(item < 0 for item in alpha) or sum(alpha) <= 0:
        raise ValueError("--reward-alpha values must be non-negative with positive sum")
    total = sum(alpha)
    return [item / total for item in alpha]


def make_dataset(args, gen_method: str) -> ApplicabilityDataset:
    return ApplicabilityDataset(
        n_actions=args.n_actions,
        n_users=args.n_users,
        dim_context=args.dim_context,
        n_cat_dim=args.n_cat_dim,
        n_cat_per_dim=args.n_cat_per_dim,
        n_unobserved_cat_dim=args.n_unobserved_cat_dim,
        n_clusters=args.n_clusters,
        gen_clustering_method=gen_method,
        cluster_balance="natural",
        cluster_temperature=args.temperature,
        alpha=parse_alpha(args.reward_alpha),
        feature_nonlinearity=args.feature_nonlinearity,
        reward_scale=args.reward_scale,
        reward_std=args.reward_std,
        beta=args.beta,
        target_epsilon=args.eps,
        deficient_action_fraction=0.0,
        random_state=args.environment_seed,
    )


def prepare_seed(args, gen_method: str, seed_index: int) -> Dict:
    bandit_data = make_dataset(args, gen_method).generate(
        n_rounds=args.n_rounds,
        sample_random_state=args.sample_seed + seed_index,
    )
    pi_e = bandit_data["target_policy_population"][
        bandit_data["user_idx"]
    ][:, :, np.newaxis]
    model_seed = args.model_seed + seed_index
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    q_x_a = fit_action_reward_model(bandit_data, model_seed)
    baselines = estimate_shared_baselines(bandit_data, pi_e, q_x_a)
    return {
        "bandit_data": bandit_data,
        "pi_e": pi_e,
        "q_x_a": q_x_a,
        "model_seed": model_seed,
        "baselines": baselines,
    }


def fit_method(args, bandit_data: Dict, method: str, seed_index: int):
    features = bandit_data["action_context_one_hot"]
    cluster_seed = args.cluster_seed + seed_index
    if method in IMPRECISE_METHODS:
        return fit_imprecise_clustering(
            action_features=features,
            n_clusters=args.n_clusters,
            method=method,
            random_state=cluster_seed,
            fuzzifier=args.fuzzifier,
            max_iter=args.cluster_max_iter,
            tolerance=args.cluster_tolerance,
            rough_ratio=args.rough_ratio,
        )
    return fixed_partition_result(
        action_features=features,
        n_clusters=args.n_clusters,
        method=method,
        random_state=cluster_seed,
        matched_labels=bandit_data["cluster_indices"],
        temperature=args.temperature,
    )


def evaluate_partition(
    bandit_data: Dict,
    pi_e: np.ndarray,
    q_x_a: np.ndarray,
    labels: np.ndarray,
    label: str,
    model_seed: int,
    include_prediction: bool = False,
) -> Dict:
    clusters_3d = clusters_to_onehot_3d(labels, bandit_data["n_users"])
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    f_x_a = fit_two_stage_model(
        bandit_data=bandit_data,
        clusters_3d=clusters_3d,
        random_state=model_seed,
    )
    estimates = estimate_partition_methods(
        bandit_data=bandit_data,
        pi_e=pi_e,
        clusters_3d=clusters_3d,
        f_x_a=f_x_a,
        q_x_a=q_x_a,
        label=label,
    )
    diagnostics = reward_diagnostics(
        prediction_2s=f_x_a[:, :, 0],
        prediction_1s=q_x_a[:, :, 0],
        expected_reward=bandit_data["expected_reward"],
        clusters=labels,
        pi_b=bandit_data["pi_b"][:, :, 0],
        pi_e=pi_e[:, :, 0],
        reward_variance=float(
            bandit_data["fixed_expected_rewards"].var()
        ),
    )
    result = {
        "estimates": estimates,
        "reward_diagnostics": diagnostics,
        "ari": float(
            adjusted_rand_score(
                bandit_data["cluster_indices"],
                labels,
            )
        ),
        "n_effective_clusters": int(np.unique(labels).size),
    }
    if include_prediction:
        result["prediction_2s"] = f_x_a[:, :, 0]
    return result


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w") as file:
            json.dump(_jsonable(value), file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def summarize_direction1(records: Iterable[Dict], output_dir: Path):
    records = list(records)
    rows = []
    baseline_rows = []
    for gen_method in sorted({record["gen_method"] for record in records}):
        selected = [
            record for record in records if record["gen_method"] == gen_method
        ]
        true_values = np.array(
            [record["true_policy_value"] for record in selected]
        )
        for baseline in ("DR", "DM", "MIPS-embedding", "IPS", "SNIPS"):
            values = np.array(
                [record["baselines"][baseline] for record in selected]
            )
            baseline_rows.append(
                {
                    "gen_method": gen_method,
                    "estimator": baseline,
                    **_error_metrics(values, true_values),
                }
            )
        methods = sorted(
            {
                method["method"]
                for record in selected
                for method in record["methods"]
                if "error" not in method
            }
        )
        for method in methods:
            paired = []
            for record in selected:
                item = next(
                    (
                        candidate
                        for candidate in record["methods"]
                        if candidate["method"] == method
                        and "error" not in candidate
                    ),
                    None,
                )
                if item is not None:
                    paired.append((record, item))
            if not paired:
                continue
            method_records = [item for _, item in paired]
            method_true_values = np.array(
                [record["true_policy_value"] for record, _ in paired]
            )
            estimator = f"OffCEM-2s::{method}"
            values = np.array(
                [item["evaluation"]["estimates"][estimator] for item in method_records]
            )
            rows.append(
                {
                    "gen_method": gen_method,
                    "method": method,
                    "n_seeds": len(method_records),
                    **_error_metrics(values, method_true_values),
                    **_mean_diagnostics(method_records),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "method_summary.csv", index=False
    )
    pd.DataFrame(baseline_rows).to_csv(
        output_dir / "baseline_summary.csv", index=False
    )
    write_json_atomic(
        output_dir / "summary.json",
        {"methods": rows, "baselines": baseline_rows},
    )


def summarize_direction6(records: Iterable[Dict], output_dir: Path):
    records = list(records)
    draw_rows = []
    seed_rows = []
    baseline_rows = []
    for gen_method in sorted({record["gen_method"] for record in records}):
        selected = [
            record for record in records if record["gen_method"] == gen_method
        ]
        true_values = np.array(
            [record["true_policy_value"] for record in selected]
        )
        for baseline in ("DR", "DM", "MIPS-embedding", "IPS", "SNIPS"):
            values = np.array(
                [record["baselines"][baseline] for record in selected]
            )
            baseline_rows.append(
                {
                    "gen_method": gen_method,
                    "estimator": baseline,
                    **_error_metrics(values, true_values),
                }
            )
    for record in records:
        true_value = record["true_policy_value"]
        for method in record["methods"]:
            if "error" in method:
                continue
            estimates = np.array(
                [
                    draw["estimate"]
                    for draw in method["partitions"]
                    if "error" not in draw
                ]
            )
            dm_values = np.array(
                [
                    draw["dm_2s"]
                    for draw in method["partitions"]
                    if "error" not in draw
                ]
            )
            if estimates.size == 0:
                continue
            for draw in method["partitions"]:
                if "error" in draw:
                    continue
                draw_rows.append(
                    {
                        "gen_method": record["gen_method"],
                        "seed": record["seed"],
                        "method": method["method"],
                        "draw": draw["draw"],
                        "estimate": draw["estimate"],
                        "relative_squared_error": (
                            (draw["estimate"] - true_value) / true_value
                        )
                        ** 2,
                        "dm_2s": draw["dm_2s"],
                        "dm_ratio": draw["dm_ratio"],
                    }
                )
            seed_rows.append(
                {
                    "gen_method": record["gen_method"],
                    "seed": record["seed"],
                    "method": method["method"],
                    "n_partitions": len(estimates),
                    "estimate_median": float(np.median(estimates)),
                    "estimate_q05": float(np.quantile(estimates, 0.05)),
                    "estimate_q95": float(np.quantile(estimates, 0.95)),
                    "estimate_range": float(estimates.max() - estimates.min()),
                    "dm_2s_mean": float(dm_values.mean()),
                    "dm_2s_q95": float(np.quantile(dm_values, 0.95)),
                    "true_policy_value": true_value,
                }
            )
    draws = pd.DataFrame(draw_rows)
    seeds = pd.DataFrame(seed_rows)
    pooled_rows = []
    for (gen_method, method), group in draws.groupby(
        ["gen_method", "method"]
    ):
        hard = group.loc[group["draw"] == 0]
        pooled_rows.append(
            {
                "gen_method": gen_method,
                "method": method,
                "hard_relMSE": float(
                    hard["relative_squared_error"].mean()
                ),
                "hard_dm_2s_mean": float(hard["dm_2s"].mean()),
                "sensitivity_relMSE": float(
                    group["relative_squared_error"].mean()
                ),
                "sensitivity_dm_2s_mean": float(group["dm_2s"].mean()),
                "sensitivity_dm_2s_std": float(group["dm_2s"].std()),
                "n_seed_partition_pairs": int(len(group)),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    draws.to_csv(output_dir / "partition_draws.csv", index=False)
    seeds.to_csv(output_dir / "seed_sensitivity_intervals.csv", index=False)
    pooled.to_csv(output_dir / "method_summary.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(
        output_dir / "baseline_summary.csv", index=False
    )
    write_json_atomic(
        output_dir / "summary.json",
        {
            "methods": pooled.to_dict(orient="records"),
            "seed_intervals": seeds.to_dict(orient="records"),
            "baselines": baseline_rows,
        },
    )


def load_records(
    output_dir: Path,
    checkpoint_names=None,
) -> List[Dict]:
    records = []
    for path in sorted((output_dir / "checkpoints").glob("*.json")):
        if checkpoint_names is not None and path.name not in checkpoint_names:
            continue
        with open(path) as file:
            records.append(json.load(file))
    return records


def _error_metrics(values, true_values):
    normalized = (values - true_values) / true_values
    return {
        "estimate_mean": float(values.mean()),
        "true_value_mean": float(true_values.mean()),
        "relMSE": float(np.mean(normalized**2)),
        "relBias2": float(np.mean(normalized) ** 2),
        "relVar": float(np.var(normalized)),
    }


def _mean_diagnostics(method_records):
    keys = method_records[0]["evaluation"]["reward_diagnostics"]
    result = {
        f"{key}_mean": float(
            np.mean(
                [
                    record["evaluation"]["reward_diagnostics"][key]
                    for record in method_records
                ]
            )
        )
        for key in keys
    }
    result["ari_mean"] = float(
        np.mean([record["evaluation"]["ari"] for record in method_records])
    )
    uncertainty_keys = sorted(
        {
            key
            for record in method_records
            for key in record["clustering"]["uncertainty"]
        }
    )
    for key in uncertainty_keys:
        values = [
            record["clustering"]["uncertainty"].get(key)
            for record in method_records
        ]
        finite = [value for value in values if value is not None]
        if finite:
            result[f"uncertainty_{key}_mean"] = float(np.mean(finite))
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
