"""Controlled test of whether FCM uncertainty adds value for OffCEM."""
import argparse
from pathlib import Path
import traceback
import warnings

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPS
from sklearn.exceptions import ConvergenceWarning
import torch
from tqdm import tqdm

from applicability.estimators import fit_action_reward_model
from imprecise.clustering import ImpreciseClusteringResult
from imprecise.clustering import fit_imprecise_clustering
from imprecise.clustering import sample_partitions
from imprecise.controlled_overlap import ControlledOverlapDataset
from imprecise.controlled_overlap import cluster_size_diagnostics
from imprecise.controlled_overlap import membership_diagnostics
from imprecise.controlled_overlap import summarize_controlled_records
from imprecise.experiment import evaluate_partition
from imprecise.experiment import load_records
from imprecise.experiment import write_json_atomic
from ope import OffPolicyEvaluation


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled reward-relevant FCM overlap experiment"
    )
    parser.add_argument(
        "--reward-modes",
        default="reward_relevant,feature_only",
    )
    parser.add_argument(
        "--ambiguity-fractions",
        default="0.0,0.25,0.5,0.75",
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
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-partition-samples", type=int, default=10)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--cluster-max-iter", type=int, default=150)
    parser.add_argument("--cluster-tolerance", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=3.0)
    parser.add_argument("--reward-std", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=-0.1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--environment-seed", type=int, default=31000)
    parser.add_argument("--sample-seed", type=int, default=41000)
    parser.add_argument("--model-seed", type=int, default=51000)
    parser.add_argument("--cluster-seed", type=int, default=61000)
    parser.add_argument(
        "--out-dir",
        default=(
            "/Users/cispa/Documents/OffCEM/imprecise_results/"
            "controlled_fcm_overlap"
        ),
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.reward_modes = "reward_relevant,feature_only"
        args.ambiguity_fractions = "0.0,0.5"
        args.n_actions = 60
        args.n_users = 30
        args.n_clusters = 4
        args.dim_context = 5
        args.dim_action = 4
        args.n_rounds = 600
        args.n_seeds = 1
        args.n_partition_samples = 2
        args.reward_std = 1.0
        args.out_dir = str(Path(args.out_dir) / "quick")

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    modes = _split_csv(args.reward_modes)
    ambiguity_levels = [
        float(value) for value in _split_csv(args.ambiguity_fractions)
    ]
    cells = [
        (mode, ambiguity, seed)
        for mode in modes
        for ambiguity in ambiguity_levels
        for seed in range(args.n_seeds)
    ]

    if not args.analyze_only:
        with tqdm(cells, desc="Controlled FCM", unit="fit") as progress:
            for reward_mode, ambiguity_fraction, seed in progress:
                progress.set_postfix_str(
                    f"{reward_mode} ambiguity={ambiguity_fraction:g} seed={seed}"
                )
                checkpoint = checkpoint_dir / (
                    f"mode-{reward_mode}_ambiguity-{ambiguity_fraction:g}_"
                    f"seed{seed:04d}.json"
                )
                if checkpoint.exists():
                    continue
                record = _run_cell(
                    args,
                    reward_mode=reward_mode,
                    ambiguity_fraction=ambiguity_fraction,
                    seed=seed,
                )
                write_json_atomic(checkpoint, record)

    records = load_records(output)
    if not records:
        raise SystemExit(f"No checkpoints found in {output}")
    summary = summarize_controlled_records(records, output)
    write_json_atomic(output / "analysis_summary.json", summary)
    failures = sum("error" in record for record in records)
    failures += sum(
        "error" in item
        for record in records
        for item in record.get("partitions", [])
    )
    print(
        f"wrote controlled FCM analysis from {len(records)} fits; "
        f"failures={failures}"
    )


def _run_cell(args, reward_mode, ambiguity_fraction, seed):
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
            reward_mode=reward_mode,
            cluster_reward_share=args.cluster_reward_share,
            reward_scale=args.reward_scale,
            reward_std=args.reward_std,
            beta=args.beta,
            target_epsilon=args.eps,
            random_state=environment_seed,
        ).generate(args.n_rounds, sample_seed)
        pi_e = data["target_policy_population"][data["user_idx"]][
            :, :, np.newaxis
        ]
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        q_x_a = fit_action_reward_model(data, model_seed)
        baselines = _estimate_baselines(data, pi_e, q_x_a)
        fcm = fit_imprecise_clustering(
            action_features=data["action_context_one_hot"],
            n_clusters=args.n_clusters,
            method="fcm",
            random_state=cluster_seed,
            fuzzifier=args.fuzzifier,
            max_iter=args.cluster_max_iter,
            tolerance=args.cluster_tolerance,
        )
        calibration = membership_diagnostics(
            estimated=fcm.scores,
            truth=data["true_memberships"],
            reward_mixture_effect=data["reward_mixture_effect"],
        )
        partitions = [
            ("fcm_hard", 0, fcm.labels),
            ("oracle_primary", 0, data["primary_clusters"]),
        ]
        partitions.extend(
            (
                "fcm_sample",
                draw,
                labels,
            )
            for draw, labels in enumerate(
                sample_partitions(
                    fcm,
                    n_samples=args.n_partition_samples,
                    random_state=cluster_seed + 10000,
                ),
                start=1,
            )
        )
        true_result = ImpreciseClusteringResult(
            method="fcm",
            labels=data["primary_clusters"],
            scores=data["true_memberships"],
            uncertainty={},
            metadata={},
        )
        partitions.extend(
            (
                "true_membership_sample",
                draw,
                labels,
            )
            for draw, labels in enumerate(
                sample_partitions(
                    true_result,
                    n_samples=args.n_partition_samples,
                    random_state=cluster_seed + 20000,
                ),
                start=1,
            )
        )

        partition_records = []
        for source, draw, labels in partitions:
            label = f"{source}::{draw}"
            try:
                evaluation = evaluate_partition(
                    bandit_data=data,
                    pi_e=pi_e,
                    q_x_a=q_x_a,
                    labels=labels,
                    label=label,
                    model_seed=model_seed,
                )
                estimator = f"OffCEM-2s::{label}"
                partition_records.append(
                    {
                        "source": source,
                        "draw": draw,
                        "estimate": evaluation["estimates"][estimator],
                        "diagnostics": evaluation["reward_diagnostics"],
                        "partition_diagnostics": {
                            "ari_primary": evaluation["ari"],
                            **cluster_size_diagnostics(
                                labels, args.n_clusters
                            ),
                        },
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
            "reward_mode": reward_mode,
            "ambiguity_fraction": ambiguity_fraction,
            "seed": seed,
            "true_policy_value": data["true_policy_value"],
            "baselines": baselines,
            "membership_diagnostics": calibration,
            "fcm_uncertainty": fcm.uncertainty,
            "partitions": partition_records,
        }
    except Exception as error:
        return {
            "reward_mode": reward_mode,
            "ambiguity_fraction": ambiguity_fraction,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "partitions": [],
        }


def _estimate_baselines(data, pi_e, q_x_a):
    estimators = [
        IPS(estimator_name="IPS"),
        SNIPS(estimator_name="SNIPS"),
        DR(estimator_name="DR"),
        DM(estimator_name="DM"),
    ]
    ope = OffPolicyEvaluation(
        bandit_feedback=data,
        ope_estimators=estimators,
    )
    return ope.estimate_policy_values(
        action_dist=pi_e,
        estimated_rewards_by_reg_model={
            "DR": q_x_a,
            "DM": q_x_a,
        },
    )


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
