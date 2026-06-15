"""Compare feature, domain, and outcome-aware partitions for OffCEM."""
import argparse
import json
from pathlib import Path
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import torch
from tqdm import tqdm

from applicability.estimators import estimate_shared_baselines
from applicability.estimators import fit_action_reward_model
from imprecise.domain_outcome import build_partitions
from imprecise.domain_outcome import DomainOutcomeDataset
from imprecise.domain_outcome import partition_diagnostics
from imprecise.domain_outcome import summarize_domain_outcome
from imprecise.experiment import evaluate_partition
from imprecise.experiment import load_records
from imprecise.experiment import write_json_atomic
from imprecise.task_execution import add_task_arguments
from imprecise.task_execution import checkpoint_summary
from imprecise.task_execution import print_checkpoint_summary
from imprecise.task_execution import selected_task_index
from imprecise.task_execution import validate_task_arguments


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain-informed and outcome-aware OffCEM experiment"
    )
    parser.add_argument(
        "--domain-alignments",
        default="0.0,0.5,1.0",
    )
    parser.add_argument(
        "--domain-label-noise",
        default="0.0,0.25,0.5",
    )
    parser.add_argument(
        "--methods",
        default=(
            "feature_only,domain_informed,oracle_outcome,"
            "estimated_outcome,hybrid,shuffled_domain,random"
        ),
    )
    parser.add_argument("--n-actions", type=int, default=200)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--dim-context", type=int, default=8)
    parser.add_argument("--dim-generic", type=int, default=8)
    parser.add_argument("--dim-domain", type=int, default=8)
    parser.add_argument("--action-noise", type=float, default=0.2)
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--aux-rounds", type=int, default=3000)
    parser.add_argument("--profile-components", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--reward-scale", type=float, default=3.0)
    parser.add_argument("--reward-std", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-0.1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--environment-seed", type=int, default=71000)
    parser.add_argument("--sample-seed", type=int, default=72000)
    parser.add_argument("--aux-sample-seed", type=int, default=73000)
    parser.add_argument("--model-seed", type=int, default=74000)
    parser.add_argument("--cluster-seed", type=int, default=75000)
    parser.add_argument(
        "--out-dir",
        default=(
            "/Users/cispa/Documents/OffCEM/imprecise_results/"
            "domain_outcome_clustering"
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
        args.domain_alignments = "0.0,1.0"
        args.domain_label_noise = "0.0"
        args.methods = (
            "feature_only,domain_informed,oracle_outcome,"
            "estimated_outcome,hybrid,shuffled_domain,random"
        )
        args.n_actions = 60
        args.n_users = 30
        args.n_clusters = 4
        args.dim_context = 5
        args.dim_generic = 4
        args.dim_domain = 4
        args.n_rounds = 600
        args.aux_rounds = 600
        args.profile_components = 4
        args.n_seeds = 1
        args.reward_std = 1.0
        args.out_dir = str(Path(args.out_dir) / "quick")

    alignments = _float_csv(args.domain_alignments)
    noise_levels = _float_csv(args.domain_label_noise)
    methods = _split_csv(args.methods)
    cells = [
        (alignment, noise, seed)
        for alignment in alignments
        for noise in noise_levels
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
            methods,
            task_label=f"task {task_index}/{len(cells) - 1}",
        )
        path = checkpoint_paths[task_index]
        with open(path) as file:
            record = json.load(file)
        if _failure_count(record):
            raise SystemExit(1)
        return

    if not args.analyze_only:
        _run_tasks(args, cells, checkpoint_dir, methods)

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
    summary = summarize_domain_outcome(records, output)
    write_json_atomic(output / "analysis_summary.json", summary)
    failures = sum(_failure_count(record) for record in records)
    print(
        f"wrote domain/outcome analysis from {len(records)} cells; "
        f"failures={failures}"
    )


def _run_tasks(args, cells, checkpoint_dir, methods, task_label=None):
    description = "Domain/outcome"
    if task_label:
        description = f"{description} ({task_label})"
    with tqdm(cells, desc=description, unit="cell") as progress:
        for alignment, noise, seed in progress:
            progress.set_postfix_str(
                f"alignment={alignment:g} noise={noise:g} seed={seed}"
            )
            checkpoint = checkpoint_dir / _checkpoint_name(
                alignment, noise, seed
            )
            if checkpoint.exists():
                continue
            record = _run_cell(
                args,
                domain_alignment=alignment,
                domain_label_noise=noise,
                seed=seed,
                methods=methods,
            )
            write_json_atomic(checkpoint, record)


def _checkpoint_name(alignment, noise, seed):
    return (
        f"alignment-{alignment:g}_noise-{noise:g}_"
        f"seed{seed:04d}.json"
    )


def _failure_count(record):
    return int("error" in record) + sum(
        "error" in method for method in record.get("methods", [])
    )


def _run_cell(
    args,
    domain_alignment,
    domain_label_noise,
    seed,
    methods,
):
    dataset = DomainOutcomeDataset(
        n_actions=args.n_actions,
        n_users=args.n_users,
        n_clusters=args.n_clusters,
        dim_context=args.dim_context,
        dim_generic=args.dim_generic,
        dim_domain=args.dim_domain,
        domain_alignment=domain_alignment,
        domain_label_noise=domain_label_noise,
        action_noise=args.action_noise,
        reward_scale=args.reward_scale,
        reward_std=args.reward_std,
        beta=args.beta,
        target_epsilon=args.eps,
        random_state=args.environment_seed + seed,
    )
    try:
        evaluation_data = dataset.generate(
            n_rounds=args.n_rounds,
            sample_random_state=args.sample_seed + seed,
        )
        auxiliary_data = dataset.generate(
            n_rounds=args.aux_rounds,
            sample_random_state=args.aux_sample_seed + seed,
        )
        model_seed = args.model_seed + seed
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        evaluation_q = fit_action_reward_model(
            evaluation_data, model_seed
        )
        auxiliary_q = fit_action_reward_model(
            auxiliary_data, model_seed + 10000
        )
        pi_e = evaluation_data["target_policy_population"][
            evaluation_data["user_idx"]
        ][:, :, np.newaxis]
        baselines = estimate_shared_baselines(
            evaluation_data, pi_e, evaluation_q
        )
        all_partitions = build_partitions(
            evaluation_data=evaluation_data,
            auxiliary_q=auxiliary_q,
            n_clusters=args.n_clusters,
            random_state=args.cluster_seed + seed,
            profile_components=args.profile_components,
        )
        method_records = []
        for method in methods:
            if method not in all_partitions:
                method_records.append(
                    {
                        "method": method,
                        "error": f"Unknown method: {method}",
                    }
                )
                continue
            labels = all_partitions[method]
            try:
                evaluation = evaluate_partition(
                    bandit_data=evaluation_data,
                    pi_e=pi_e,
                    q_x_a=evaluation_q,
                    labels=labels,
                    label=method,
                    model_seed=model_seed,
                )
                estimator = f"OffCEM-2s::{method}"
                method_records.append(
                    {
                        "method": method,
                        "estimate": evaluation["estimates"][estimator],
                        "diagnostics": evaluation[
                            "reward_diagnostics"
                        ],
                        "partition_diagnostics": partition_diagnostics(
                            labels, evaluation_data
                        ),
                    }
                )
            except Exception as error:
                method_records.append(
                    {
                        "method": method,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
        return {
            "domain_alignment": domain_alignment,
            "domain_label_noise": domain_label_noise,
            "seed": seed,
            "true_policy_value": evaluation_data["true_policy_value"],
            "baselines": baselines,
            "methods": method_records,
            "auxiliary_is_independent": True,
        }
    except Exception as error:
        return {
            "domain_alignment": domain_alignment,
            "domain_label_noise": domain_label_noise,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "methods": [],
        }


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_csv(value):
    return [float(item) for item in _split_csv(value)]


if __name__ == "__main__":
    main()
