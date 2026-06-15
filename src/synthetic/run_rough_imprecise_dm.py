"""Partial-identification experiment for rough-set DM extrapolation bounds."""
import argparse
import json
from pathlib import Path
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import torch
from tqdm import tqdm

from applicability.dataset import ApplicabilityDataset
from applicability.estimators import fit_action_reward_model
from imprecise.clustering import fit_imprecise_clustering
from imprecise.experiment import load_records
from imprecise.experiment import parse_alpha
from imprecise.experiment import write_json_atomic
from imprecise.rough_imprecise_dm import rough_dm_bounds
from imprecise.rough_imprecise_dm import summarize_rough_dm_records
from imprecise.task_execution import add_task_arguments
from imprecise.task_execution import checkpoint_summary
from imprecise.task_execution import print_checkpoint_summary
from imprecise.task_execution import selected_task_index
from imprecise.task_execution import validate_task_arguments


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rough-set DM bounds under deficient action support"
    )
    parser.add_argument(
        "--deficient-action-fractions",
        default="0.1,0.3,0.5",
    )
    parser.add_argument(
        "--support-modes",
        default="action,contextual",
        help="Comma-separated action-global and/or contextual support",
    )
    parser.add_argument(
        "--rough-ratios",
        default="1.05,1.15,1.25,1.5",
    )
    parser.add_argument("--gammas", default="0.0,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--calibration-quantile", type=float, default=0.9)
    parser.add_argument("--min-calibration-count", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--outcome-lower", type=float, default=-15.0)
    parser.add_argument("--outcome-upper", type=float, default=15.0)
    parser.add_argument("--n-actions", type=int, default=200)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--dim-context", type=int, default=8)
    parser.add_argument("--n-cat-dim", type=int, default=8)
    parser.add_argument("--n-cat-per-dim", type=int, default=5)
    parser.add_argument("--n-unobserved-cat-dim", type=int, default=2)
    parser.add_argument("--n-rounds", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--gen-clustering-method",
        default="feature_bucket",
    )
    parser.add_argument(
        "--reward-alpha",
        default="0.4,0.3,0.2,0.1",
    )
    parser.add_argument("--feature-nonlinearity", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--reward-scale", type=float, default=3.0)
    parser.add_argument("--reward-std", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-0.1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--cluster-max-iter", type=int, default=150)
    parser.add_argument("--cluster-tolerance", type=float, default=1e-5)
    parser.add_argument("--environment-seed", type=int, default=81000)
    parser.add_argument("--sample-seed", type=int, default=82000)
    parser.add_argument("--model-seed", type=int, default=83000)
    parser.add_argument("--cluster-seed", type=int, default=84000)
    parser.add_argument(
        "--out-dir",
        default=(
            "/Users/cispa/Documents/OffCEM/imprecise_results/"
            "rough_imprecise_dm"
        ),
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    add_task_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_task_arguments(args)
    if args.quick:
        args.deficient_action_fractions = "0.3"
        args.support_modes = "action,contextual"
        args.rough_ratios = "1.15"
        args.gammas = "0.0,1.0,2.0,4.0,8.0"
        args.n_actions = 60
        args.n_users = 30
        args.n_clusters = 4
        args.dim_context = 5
        args.n_cat_dim = 5
        args.n_cat_per_dim = 4
        args.n_unobserved_cat_dim = 1
        args.n_rounds = 600
        args.n_seeds = 1
        args.min_calibration_count = 5
        args.out_dir = str(Path(args.out_dir) / "quick")

    deficient_fractions = _float_csv(args.deficient_action_fractions)
    rough_ratios = _float_csv(args.rough_ratios)
    support_modes = _split_csv(args.support_modes)
    invalid_modes = set(support_modes) - {"action", "contextual"}
    if invalid_modes:
        raise SystemExit(
            f"Unknown support modes: {sorted(invalid_modes)}"
        )
    cells = [
        (support_mode, deficient_fraction, rough_ratio, seed)
        for support_mode in support_modes
        for deficient_fraction in deficient_fractions
        for rough_ratio in rough_ratios
        for seed in range(args.n_seeds)
    ]
    if args.print_task_count:
        print(len(cells))
        return

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [
        checkpoint_dir / _checkpoint_name(*cell)
        for cell in cells
    ]
    task_index = selected_task_index(args, len(cells))
    if args.status_only:
        print_checkpoint_summary(
            checkpoint_summary(checkpoint_paths, _failure_count)
        )
        return

    if task_index is not None:
        _run_tasks(
            args,
            [cells[task_index]],
            checkpoint_dir,
            task_label=f"task {task_index}/{len(cells) - 1}",
        )
        with open(checkpoint_paths[task_index]) as file:
            record = json.load(file)
        if _failure_count(record):
            raise SystemExit(1)
        return

    if not args.analyze_only:
        _run_tasks(args, cells, checkpoint_dir)

    status = checkpoint_summary(checkpoint_paths, _failure_count)
    print_checkpoint_summary(status)
    if not args.allow_incomplete and (status["missing"] or status["error"]):
        raise SystemExit(
            "Refusing to analyze incomplete results. Retry missing/failed "
            "tasks or pass --allow-incomplete."
        )
    records = load_records(
        output,
        checkpoint_names={path.name for path in checkpoint_paths},
    )
    summary = summarize_rough_dm_records(records, output)
    write_json_atomic(output / "analysis_summary.json", summary)
    print(
        f"wrote rough-DM analysis from {len(records)} fits; "
        f"failures={sum(_failure_count(record) for record in records)}"
    )


def _run_tasks(args, cells, checkpoint_dir, task_label=None):
    description = "Rough imprecise DM"
    if task_label:
        description = f"{description} ({task_label})"
    with tqdm(cells, desc=description, unit="fit") as progress:
        for support_mode, deficient_fraction, rough_ratio, seed in progress:
            progress.set_postfix_str(
                f"mode={support_mode} deficient={deficient_fraction:g} "
                f"ratio={rough_ratio:g} seed={seed}"
            )
            checkpoint = checkpoint_dir / _checkpoint_name(
                support_mode, deficient_fraction, rough_ratio, seed
            )
            if checkpoint.exists():
                continue
            write_json_atomic(
                checkpoint,
                _run_cell(
                    args,
                    support_mode=support_mode,
                    deficient_action_fraction=deficient_fraction,
                    rough_ratio=rough_ratio,
                    seed=seed,
                ),
            )


def _run_cell(
    args,
    support_mode,
    deficient_action_fraction,
    rough_ratio,
    seed,
):
    try:
        environment_seed = args.environment_seed + seed
        sample_seed = args.sample_seed + seed
        model_seed = args.model_seed + seed
        cluster_seed = args.cluster_seed + seed
        data = ApplicabilityDataset(
            n_actions=args.n_actions,
            n_users=args.n_users,
            dim_context=args.dim_context,
            n_cat_dim=args.n_cat_dim,
            n_cat_per_dim=args.n_cat_per_dim,
            n_unobserved_cat_dim=args.n_unobserved_cat_dim,
            n_clusters=args.n_clusters,
            gen_clustering_method=args.gen_clustering_method,
            cluster_balance="natural",
            cluster_temperature=args.temperature,
            alpha=parse_alpha(args.reward_alpha),
            feature_nonlinearity=args.feature_nonlinearity,
            reward_scale=args.reward_scale,
            reward_std=args.reward_std,
            beta=args.beta,
            target_epsilon=args.eps,
            deficient_action_fraction=(
                deficient_action_fraction
                if support_mode == "action"
                else 0.0
            ),
            random_state=environment_seed,
        ).generate(args.n_rounds, sample_random_state=sample_seed)
        if support_mode == "contextual":
            data = _apply_contextual_support(
                data=data,
                deficient_fraction=deficient_action_fraction,
                sample_seed=sample_seed,
                reward_std=args.reward_std,
            )
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        q_x_a = fit_action_reward_model(data, model_seed)[:, :, 0]
        pi_e = data["target_policy_population"][data["user_idx"]]
        expected_reward = data["expected_reward"]
        support = (
            data["support_mask"]
            if support_mode == "action"
            else data["support_matrix"]
        )
        standard_dm = float(np.mean(np.sum(pi_e * q_x_a, axis=1)))
        sample_true = float(
            np.mean(np.sum(pi_e * expected_reward, axis=1))
        )
        rough = fit_imprecise_clustering(
            action_features=data["action_context_one_hot"],
            n_clusters=args.n_clusters,
            method="rough",
            random_state=cluster_seed,
            max_iter=args.cluster_max_iter,
            tolerance=args.cluster_tolerance,
            rough_ratio=rough_ratio,
        )
        bounds = []
        for gamma in _float_csv(args.gammas):
            try:
                result = rough_dm_bounds(
                    prediction=q_x_a,
                    target_policy=pi_e,
                    support_mask=support,
                    action_features=data["action_context_one_hot"],
                    hard_labels=rough.labels,
                    candidates=rough.candidates,
                    factual_action=data["action"],
                    factual_reward=data["reward"],
                    gamma=gamma,
                    outcome_lower=args.outcome_lower,
                    outcome_upper=args.outcome_upper,
                    calibration_quantile=args.calibration_quantile,
                    n_neighbors=args.n_neighbors,
                    min_calibration_count=args.min_calibration_count,
                    expected_reward=expected_reward,
                )
                result.pop("lower_prediction")
                result.pop("upper_prediction")
                result.update(
                    {
                        "gamma": gamma,
                        "conditional_coverage": bool(
                            result["lower_value"]
                            <= sample_true
                            <= result["upper_value"]
                        ),
                        "population_coverage": bool(
                            result["lower_value"]
                            <= data["true_policy_value"]
                            <= result["upper_value"]
                        ),
                        "manski_conditional_coverage": bool(
                            result["manski_lower"]
                            <= sample_true
                            <= result["manski_upper"]
                        ),
                        "manski_population_coverage": bool(
                            result["manski_lower"]
                            <= data["true_policy_value"]
                            <= result["manski_upper"]
                        ),
                    }
                )
                bounds.append(result)
            except Exception as error:
                bounds.append(
                    {
                        "gamma": gamma,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
        population_reward = data["fixed_expected_rewards"]
        return {
            "support_mode": support_mode,
            "deficient_action_fraction": deficient_action_fraction,
            "rough_ratio": rough_ratio,
            "seed": seed,
            "true_policy_value": data["true_policy_value"],
            "sample_true_policy_value": sample_true,
            "standard_dm_value": standard_dm,
            "standard_dm_sample_error": (
                (standard_dm - sample_true) / max(abs(sample_true), 1e-12)
            )
            ** 2,
            "outcome_range_covers_population": bool(
                population_reward.min() >= args.outcome_lower
                and population_reward.max() <= args.outcome_upper
            ),
            "population_reward_min": float(population_reward.min()),
            "population_reward_max": float(population_reward.max()),
            "rough_uncertainty": rough.uncertainty,
            "rough_metadata": rough.metadata,
            "bounds": bounds,
        }
    except Exception as error:
        return {
            "support_mode": support_mode,
            "deficient_action_fraction": deficient_action_fraction,
            "rough_ratio": rough_ratio,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "bounds": [],
        }


def _checkpoint_name(support_mode, deficient_fraction, rough_ratio, seed):
    return (
        f"mode-{support_mode}_deficient-{deficient_fraction:g}_"
        f"ratio-{rough_ratio:g}_"
        f"seed{seed:04d}.json"
    )


def _failure_count(record):
    return int("error" in record) + sum(
        "error" in bound for bound in record.get("bounds", [])
    )


def _float_csv(value):
    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def _split_csv(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _apply_contextual_support(
    data,
    deficient_fraction,
    sample_seed,
    reward_std,
):
    """Replace the logged sample with context-dependent behavior support."""
    n_users = data["n_users"]
    n_actions = data["n_actions"]
    n_deficient = int(round(deficient_fraction * n_actions))
    if not 0 <= n_deficient < n_actions:
        raise ValueError("contextual deficiency must leave supported actions")
    support_population = np.ones((n_users, n_actions), dtype=bool)
    support_rng = np.random.RandomState(sample_seed + 7001)
    for user in range(n_users):
        if n_deficient:
            unsupported = support_rng.choice(
                n_actions, size=n_deficient, replace=False
            )
            support_population[user, unsupported] = False

    behavior = (
        data["pi_b_full_population"] * support_population
    )
    behavior /= behavior.sum(axis=1, keepdims=True)
    user_idx = data["user_idx"]
    pi_b_rows = behavior[user_idx]
    sample_rng = np.random.RandomState(sample_seed + 9001)
    actions = np.array(
        [
            sample_rng.choice(n_actions, p=probability)
            for probability in pi_b_rows
        ],
        dtype=int,
    )
    expected_reward = data["fixed_expected_rewards"][user_idx]
    factual_mean = expected_reward[np.arange(actions.size), actions]
    rewards = sample_rng.normal(factual_mean, reward_std)
    reward_mat = np.zeros((n_users, n_actions))
    obs_mat = np.zeros((n_users, n_actions), dtype=int)
    for user, action, reward in zip(user_idx, actions, rewards):
        reward_mat[user, action] = reward
        obs_mat[user, action] = 1

    data.update(
        {
            "action": actions,
            "reward": rewards,
            "pscore": pi_b_rows[np.arange(actions.size), actions],
            "pi_b": pi_b_rows[:, :, np.newaxis],
            "pi_b_population": behavior,
            "expected_reward": expected_reward,
            "action_embed": data["action_context"][actions],
            "reward_mat": reward_mat,
            "obs_mat": obs_mat,
            "support_matrix": support_population[user_idx],
        }
    )
    return data


if __name__ == "__main__":
    main()
