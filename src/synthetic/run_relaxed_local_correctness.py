"""Controlled rough-OffCEM experiment under approximate local correctness."""
import argparse
import json
from pathlib import Path
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import torch
from tqdm import tqdm

from applicability.estimators import fit_action_reward_model
from imprecise.clustering import fit_imprecise_clustering
from imprecise.clustering import sample_partitions
from imprecise.controlled_overlap import ControlledOverlapDataset
from imprecise.experiment import evaluate_partition
from imprecise.experiment import load_records
from imprecise.experiment import write_json_atomic
from imprecise.relaxed_local_correctness import (
    relaxed_local_correctness_bound,
)
from imprecise.relaxed_local_correctness import summarize_relaxed_records
from imprecise.task_execution import add_task_arguments
from imprecise.task_execution import checkpoint_summary
from imprecise.task_execution import print_checkpoint_summary
from imprecise.task_execution import selected_task_index
from imprecise.task_execution import validate_task_arguments


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test epsilon-relaxed local correctness with rough K-means "
            "partition ambiguity"
        )
    )
    parser.add_argument(
        "--ambiguity-fractions",
        default="0.0,0.25,0.5,0.75",
    )
    parser.add_argument(
        "--rough-ratios",
        default="1.05,1.15,1.25,1.5",
    )
    parser.add_argument("--ambiguity-strength", type=float, default=0.45)
    parser.add_argument("--feature-noise", type=float, default=0.10)
    parser.add_argument("--cluster-reward-share", type=float, default=0.8)
    parser.add_argument("--n-actions", type=int, default=200)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--dim-context", type=int, default=8)
    parser.add_argument("--dim-action", type=int, default=8)
    parser.add_argument("--n-rounds", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-partition-samples", type=int, default=5)
    parser.add_argument("--cluster-max-iter", type=int, default=150)
    parser.add_argument("--cluster-tolerance", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=3.0)
    parser.add_argument("--reward-std", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-0.1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--environment-seed", type=int, default=71000)
    parser.add_argument("--sample-seed", type=int, default=72000)
    parser.add_argument("--model-seed", type=int, default=73000)
    parser.add_argument("--cluster-seed", type=int, default=74000)
    parser.add_argument(
        "--out-dir",
        default=(
            "/Users/cispa/Documents/OffCEM/imprecise_results/"
            "relaxed_local_correctness"
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
        args.ambiguity_fractions = "0.0,0.5"
        args.rough_ratios = "1.1"
        args.n_actions = 60
        args.n_users = 30
        args.n_clusters = 4
        args.dim_context = 5
        args.dim_action = 4
        args.n_rounds = 600
        args.n_seeds = 1
        args.n_partition_samples = 1
        args.reward_std = 1.0
        args.out_dir = str(Path(args.out_dir) / "quick")

    ambiguity_levels = _float_csv(args.ambiguity_fractions)
    rough_ratios = _float_csv(args.rough_ratios)
    cells = [
        (ambiguity, rough_ratio, seed)
        for ambiguity in ambiguity_levels
        for rough_ratio in rough_ratios
        for seed in range(args.n_seeds)
    ]
    if args.print_task_count:
        print(len(cells))
        return

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    task_index = selected_task_index(args, len(cells))
    checkpoint_paths = [
        checkpoint_dir / _checkpoint_name(*cell)
        for cell in cells
    ]
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
        path = checkpoint_paths[task_index]
        with open(path) as file:
            record = json.load(file)
        if _failure_count(record):
            raise SystemExit(1)
        return

    if not args.analyze_only:
        _run_tasks(args, cells, checkpoint_dir)

    status = checkpoint_summary(checkpoint_paths, _failure_count)
    print_checkpoint_summary(status)
    if args.analyze_only and not args.allow_incomplete and (
        status["missing"] or status["error"]
    ):
        raise SystemExit(
            "Refusing to analyze incomplete results. Retry missing/failed "
            "tasks or pass --allow-incomplete."
        )

    records = load_records(
        output,
        checkpoint_names={path.name for path in checkpoint_paths},
    )
    if not records:
        raise SystemExit(f"No checkpoints found in {output}")
    summary = summarize_relaxed_records(records, output)
    write_json_atomic(output / "analysis_summary.json", summary)
    failures = sum(_failure_count(record) for record in records)
    print(
        f"wrote relaxed-local-correctness analysis from {len(records)} fits; "
        f"failures={failures}"
    )


def _run_tasks(args, cells, checkpoint_dir, task_label=None):
    description = "Relaxed local correctness"
    if task_label:
        description = f"{description} ({task_label})"
    with tqdm(cells, desc=description, unit="fit") as bar:
        for ambiguity, rough_ratio, seed in bar:
            bar.set_postfix_str(
                f"ambiguity={ambiguity:g} ratio={rough_ratio:g} seed={seed}"
            )
            checkpoint = checkpoint_dir / _checkpoint_name(
                ambiguity, rough_ratio, seed
            )
            if checkpoint.exists():
                continue
            record = _run_cell(
                args,
                ambiguity_fraction=ambiguity,
                rough_ratio=rough_ratio,
                seed=seed,
            )
            write_json_atomic(checkpoint, record)


def _checkpoint_name(ambiguity, rough_ratio, seed):
    return (
        f"ambiguity-{ambiguity:g}_ratio-{rough_ratio:g}_"
        f"seed{seed:04d}.json"
    )


def _failure_count(record):
    return int("error" in record) + sum(
        "error" in partition
        for partition in record.get("partitions", [])
    )


def _run_cell(args, ambiguity_fraction, rough_ratio, seed):
    environment_seed = args.environment_seed + seed
    sample_seed = args.sample_seed + seed
    model_seed = args.model_seed + seed
    cluster_seed = args.cluster_seed + seed
    try:
        data = ControlledOverlapDataset(
            n_actions=args.n_actions,
            n_users=args.n_users,
            n_clusters=args.n_clusters,
            dim_context=args.dim_context,
            dim_action=args.dim_action,
            ambiguity_fraction=ambiguity_fraction,
            ambiguity_strength=args.ambiguity_strength,
            feature_noise=args.feature_noise,
            reward_mode="reward_relevant",
            cluster_reward_share=args.cluster_reward_share,
            reward_scale=args.reward_scale,
            reward_std=args.reward_std,
            beta=args.beta,
            target_epsilon=args.eps,
            random_state=environment_seed,
        ).generate(args.n_rounds, sample_seed)
        pi_b = data["pi_b"][:, :, 0]
        pi_e = data["target_policy_population"][data["user_idx"]]
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        q_x_a = fit_action_reward_model(data, model_seed)
        rough = fit_imprecise_clustering(
            action_features=data["action_context_one_hot"],
            n_clusters=args.n_clusters,
            method="rough",
            random_state=cluster_seed,
            max_iter=args.cluster_max_iter,
            tolerance=args.cluster_tolerance,
            rough_ratio=rough_ratio,
        )
        partitions = [("rough_hard", 0, rough.labels)]
        partitions.extend(
            ("rough_sample", draw, labels)
            for draw, labels in enumerate(
                sample_partitions(
                    rough,
                    n_samples=args.n_partition_samples,
                    random_state=cluster_seed + 10000,
                ),
                start=1,
            )
        )
        partitions.append(("oracle_primary", 0, data["primary_clusters"]))

        partition_records = []
        for source, draw, labels in partitions:
            label = f"{source}::{draw}"
            try:
                evaluation = evaluate_partition(
                    bandit_data=data,
                    pi_e=pi_e[:, :, np.newaxis],
                    q_x_a=q_x_a,
                    labels=labels,
                    label=label,
                    model_seed=model_seed,
                    include_prediction=True,
                )
                estimator = f"OffCEM-2s::{label}"
                relaxed = relaxed_local_correctness_bound(
                    expected_reward=data["expected_reward"],
                    prediction=evaluation.pop("prediction_2s"),
                    pi_b=pi_b,
                    pi_e=pi_e,
                    clusters=labels,
                )
                partition_records.append(
                    {
                        "source": source,
                        "draw": draw,
                        "estimate": evaluation["estimates"][estimator],
                        "diagnostics": evaluation["reward_diagnostics"],
                        "ari_primary": evaluation["ari"],
                        "relaxed_bound": relaxed,
                    }
                )
            except Exception as error:
                partition_records.append(
                    {
                        "source": source,
                        "draw": draw,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
        return {
            "ambiguity_fraction": ambiguity_fraction,
            "rough_ratio": rough_ratio,
            "seed": seed,
            "rough_uncertainty": rough.uncertainty,
            "rough_metadata": rough.metadata,
            "partitions": partition_records,
        }
    except Exception as error:
        return {
            "ambiguity_fraction": ambiguity_fraction,
            "rough_ratio": rough_ratio,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "partitions": [],
        }


def _float_csv(value):
    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


if __name__ == "__main__":
    main()
