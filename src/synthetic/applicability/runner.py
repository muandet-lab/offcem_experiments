"""Dataset-first execution for the OffCEM applicability benchmark."""
import json
import os
from pathlib import Path
from time import time
import traceback
from typing import Dict
from typing import List
from typing import Tuple
import uuid

import numpy as np
from obp.dataset import linear_reward_function
import torch

from applicability.config import config_hash
from applicability.config import checkpoint_name
from applicability.config import task_coordinates
from applicability.dataset import ApplicabilityDataset
from applicability.diagnostics import overlap_diagnostics
from applicability.diagnostics import partition_diagnostics
from applicability.diagnostics import reward_diagnostics
from applicability.estimators import estimate_partition_methods
from applicability.estimators import estimate_shared_baselines
from applicability.estimators import fit_action_reward_model
from applicability.estimators import fit_two_stage_model
from clustering import compute_clusters
from clustering import corrupt_partition
from clustering import clusters_to_onehot_3d
from dataset import SyntheticBanditDataset
from policy import gen_eps_greedy


def run_benchmark(
    cells: List[Dict],
    output_dir: str,
    n_seeds: int,
    base_sample_seed: int,
    phase: str,
    force: bool = False,
) -> None:
    total = len(cells) * n_seeds
    failed = []
    for cell_index, cell in enumerate(cells):
        for seed_index in range(n_seeds):
            index = cell_index * n_seeds + seed_index
            print(f"[{index + 1}/{total}]", end=" ")
            result = run_task(
                cells=cells,
                output_dir=output_dir,
                n_seeds=n_seeds,
                base_sample_seed=base_sample_seed,
                phase=phase,
                cell_index=cell_index,
                seed_index=seed_index,
                force=force,
            )
            if result["status"] == "error":
                failed.append((cell_index, seed_index, result["error"]))
    if failed:
        raise RuntimeError(f"{len(failed)} benchmark tasks failed")


def run_task_index(
    cells: List[Dict],
    output_dir: str,
    n_seeds: int,
    base_sample_seed: int,
    phase: str,
    task_index: int,
    force: bool = False,
) -> Dict:
    cell_index, seed_index = task_coordinates(
        task_index=task_index,
        n_cells=len(cells),
        n_seeds=n_seeds,
    )
    return run_task(
        cells=cells,
        output_dir=output_dir,
        n_seeds=n_seeds,
        base_sample_seed=base_sample_seed,
        phase=phase,
        cell_index=cell_index,
        seed_index=seed_index,
        force=force,
    )


def run_task(
    cells: List[Dict],
    output_dir: str,
    n_seeds: int,
    base_sample_seed: int,
    phase: str,
    cell_index: int,
    seed_index: int,
    force: bool = False,
) -> Dict:
    if cell_index < 0 or cell_index >= len(cells):
        raise ValueError(
            f"cell_index must be in [0, {len(cells)}), got {cell_index}"
        )
    if seed_index < 0 or seed_index >= n_seeds:
        raise ValueError(
            f"seed_index must be in [0, {n_seeds}), got {seed_index}"
        )

    cell = cells[cell_index]
    cell_id = config_hash(cell)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / checkpoint_name(cell, seed_index)
    cached = _load_successful_checkpoint(path)
    if cached is not None and not force:
        print(
            f"cell={cell_index} seed={seed_index} id={cell_id}: cached"
        )
        return cached

    print(f"cell={cell_index} seed={seed_index} id={cell_id}: running")
    started = time()
    try:
        record = run_cell_seed(
            cell=cell,
            cell_id=cell_id,
            seed_index=seed_index,
            sample_seed=base_sample_seed + seed_index,
            phase=phase,
        )
        partition_errors = [
            partition
            for partition in record["partitions"]
            if partition.get("error")
        ]
        if partition_errors:
            labels = ", ".join(item["label"] for item in partition_errors)
            record["status"] = "error"
            record["error"] = f"Partition failures: {labels}"
        else:
            record["status"] = "ok"
    except Exception as error:
        record = {
            "status": "error",
            "cell_id": cell_id,
            "cell_index": cell_index,
            "seed_index": seed_index,
            "phase": phase,
            "cell": cell,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    record["cell_index"] = cell_index
    record["elapsed_seconds"] = time() - started
    _atomic_checkpoint(path, record)
    if record["status"] == "error":
        print(f"  ERROR: {record['error']}")
    else:
        print(f"  done in {record['elapsed_seconds']:.1f}s")
    return record


def checkpoint_summary(
    cells: List[Dict],
    output_dir: str,
    n_seeds: int,
) -> Dict:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    summary = {"expected": len(cells) * n_seeds, "ok": 0, "error": 0, "missing": 0}
    details = {"error": [], "missing": []}
    for cell_index, cell in enumerate(cells):
        for seed_index in range(n_seeds):
            path = checkpoint_dir / checkpoint_name(cell, seed_index)
            if not path.exists():
                summary["missing"] += 1
                details["missing"].append(
                    {"cell_index": cell_index, "seed_index": seed_index}
                )
                continue
            try:
                with open(path) as file:
                    record = json.load(file)
            except (OSError, json.JSONDecodeError):
                summary["error"] += 1
                details["error"].append(
                    {
                        "cell_index": cell_index,
                        "seed_index": seed_index,
                        "error": "Unreadable checkpoint",
                    }
                )
                continue
            status = record.get("status")
            if status == "ok":
                summary["ok"] += 1
            else:
                summary["error"] += 1
                details["error"].append(
                    {
                        "cell_index": cell_index,
                        "seed_index": seed_index,
                        "error": record.get("error", f"status={status}"),
                    }
                )
    summary["details"] = details
    return summary


def run_cell_seed(
    cell: Dict,
    cell_id: str,
    seed_index: int,
    sample_seed: int,
    phase: str,
) -> Dict:
    bandit_data = _generate_bandit_data(
        cell=cell,
        seed_index=seed_index,
        sample_seed=sample_seed,
    )
    pi_e = bandit_data["target_policy_population"][
        bandit_data["user_idx"]
    ][:, :, np.newaxis]

    model_seed = int(cell.get("model_seed", 24680)) + seed_index
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    q_x_a = fit_action_reward_model(bandit_data, model_seed)
    shared_estimates = estimate_shared_baselines(bandit_data, pi_e, q_x_a)

    partitions = []
    for partition_config in cell["partitions"]:
        label, clusters_1d, clusters_3d = _build_partition(
            bandit_data=bandit_data,
            config=partition_config,
            seed=int(cell.get("partition_seed", 13579)),
        )
        try:
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
            reward_diag = reward_diagnostics(
                prediction_2s=f_x_a[:, :, 0],
                prediction_1s=q_x_a[:, :, 0],
                expected_reward=bandit_data["expected_reward"],
                clusters=clusters_1d,
                pi_b=bandit_data["pi_b"][:, :, 0],
                pi_e=pi_e[:, :, 0],
                reward_variance=float(
                    bandit_data["fixed_expected_rewards"].var()
                ),
            )
            diagnostics = {
                "reward": reward_diag,
                "overlap": overlap_diagnostics(
                    bandit_data=bandit_data,
                    pi_e_rows=pi_e[:, :, 0],
                    clusters=clusters_1d,
                ),
                "partition": partition_diagnostics(
                    generating_clusters=bandit_data["cluster_indices"],
                    estimation_clusters=clusters_1d,
                ),
            }
            partitions.append(
                {
                    "label": label,
                    "config": partition_config,
                    "estimates": estimates,
                    "diagnostics": diagnostics,
                }
            )
        except Exception as error:
            partitions.append(
                {
                    "label": label,
                    "config": partition_config,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "cell_id": cell_id,
        "seed_index": seed_index,
        "sample_seed": sample_seed,
        "phase": phase,
        "cell": cell,
        "true_policy_value": bandit_data["true_policy_value"],
        "nominal_variance_shares": bandit_data.get(
            "nominal_variance_shares", {}
        ),
        "realized_variance_shares": bandit_data.get(
            "realized_variance_shares", {}
        ),
        "shared_estimates": shared_estimates,
        "partitions": partitions,
    }


def _generate_bandit_data(
    cell: Dict,
    seed_index: int,
    sample_seed: int,
) -> Dict:
    if cell.get("generator", "applicability") == "legacy_figure7":
        return _generate_legacy_figure7(cell, seed_index)

    dataset_config = dict(cell["dataset"])
    reward_config = dict(cell["reward"])
    dataset = ApplicabilityDataset(
        n_actions=int(dataset_config["n_actions"]),
        n_users=int(dataset_config["n_users"]),
        dim_context=int(dataset_config["dim_context"]),
        n_cat_dim=int(dataset_config["n_cat_dim"]),
        n_cat_per_dim=int(dataset_config["n_cat_per_dim"]),
        n_unobserved_cat_dim=int(dataset_config["n_unobserved_cat_dim"]),
        n_clusters=int(dataset_config["n_clusters"]),
        gen_clustering_method=dataset_config["gen_clustering_method"],
        cluster_balance=dataset_config["cluster_balance"],
        cluster_temperature=float(dataset_config["cluster_temperature"]),
        alpha=reward_config["alpha"],
        feature_nonlinearity=float(reward_config["feature_nonlinearity"]),
        reward_scale=float(reward_config["scale"]),
        reward_std=float(reward_config["noise_std"]),
        beta=float(cell["policies"]["behavior_beta"]),
        target_epsilon=float(cell["policies"]["target_epsilon"]),
        deficient_action_fraction=float(
            cell["policies"]["deficient_action_fraction"]
        ),
        random_state=int(cell.get("environment_seed", 12345)),
    )
    return dataset.generate(
        n_rounds=int(cell["n_rounds"]),
        sample_random_state=sample_seed,
    )


def _generate_legacy_figure7(cell: Dict, seed_index: int) -> Dict:
    """Recreate the sequential data stream used by ``run_figure7.py``.

    Reconstructing and advancing the dataset makes every checkpoint independent
    while retaining the original runner's per-seed data sequence.
    """
    dataset_config = cell["dataset"]
    reward_config = cell["reward"]
    policies = cell["policies"]
    random_state = int(cell.get("environment_seed", 12345))
    dataset = SyntheticBanditDataset(
        n_actions=int(dataset_config["n_actions"]),
        dim_context=int(dataset_config["dim_context"]),
        beta=float(policies["behavior_beta"]),
        reward_type="continuous",
        n_cat_per_dim=int(dataset_config["n_cat_per_dim"]),
        latent_param_mat_dim=int(dataset_config["latent_param_mat_dim"]),
        n_cat_dim=int(dataset_config["n_cat_dim"]),
        n_deficient_actions=0,
        reward_function=linear_reward_function,
        reward_std=float(reward_config["noise_std"]),
        random_state=random_state,
    )
    test_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=int(cell.get("n_test_rounds", 100000)),
        n_users=int(cell.get("n_test_users", 1000)),
    )
    test_policy = gen_eps_greedy(
        expected_reward=test_data["expected_reward"],
        eps=float(policies["target_epsilon"]),
    )[:, :, 0]
    true_policy_value = float(
        np.average(
            test_data["expected_reward"],
            weights=test_policy,
            axis=1,
        ).mean()
    )

    bandit_data = None
    for _ in range(seed_index + 1):
        bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=int(cell["n_rounds"]),
            n_users=int(dataset_config["n_users"]),
            n_clusters=int(dataset_config["n_clusters"]),
        )
    population_q = bandit_data["fixed_expected_rewards"]
    target_population = gen_eps_greedy(
        expected_reward=population_q,
        eps=float(policies["target_epsilon"]),
    )[:, :, 0]
    shifted = policies["behavior_beta"] * population_q
    shifted -= shifted.max(axis=1, keepdims=True)
    behavior_population = np.exp(shifted)
    behavior_population /= behavior_population.sum(axis=1, keepdims=True)
    bandit_data.update(
        target_policy_population=target_population,
        pi_b_population=behavior_population,
        support_mask=np.ones(bandit_data["n_actions"], dtype=bool),
        true_policy_value=true_policy_value,
    )
    return bandit_data


def _build_partition(
    bandit_data: Dict,
    config: Dict,
    seed: int,
) -> Tuple[str, np.ndarray, np.ndarray]:
    method = config["method"]
    if method == "matched":
        clusters_1d = bandit_data["cluster_indices"].copy()
        label = config.get("label", "matched")
    elif method == "corrupt":
        probability = float(config["probability"])
        clusters_1d = corrupt_partition(
            bandit_data["cluster_indices"], probability, seed
        )
        label = config.get("label", f"corrupt_p{probability:g}")
    else:
        clusters_1d = compute_clusters(
            action_features=bandit_data["action_context_one_hot"],
            n_clusters=int(
                config.get(
                    "n_clusters",
                    int(bandit_data["cluster_indices"].max()) + 1,
                )
            ),
            method=method,
            balance=config.get("balance", "natural"),
            random_state=seed,
            temperature=float(config.get("temperature", 10.0)),
        )
        label = config.get("label", method)
    clusters_3d = clusters_to_onehot_3d(
        clusters_1d, bandit_data["n_users"]
    )
    return label, clusters_1d, clusters_3d


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


def _load_successful_checkpoint(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as file:
            record = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return record if record.get("status") == "ok" else None


def _atomic_checkpoint(path: Path, record: Dict):
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "w") as file:
            json.dump(_jsonable(record), file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
