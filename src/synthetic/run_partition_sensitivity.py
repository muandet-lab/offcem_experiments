"""Direction 6: partition-sensitivity analysis for hard OffCEM.

For each uncertain clustering, fit one hardened partition and sample plausible
partitions from memberships, typicalities, or rough boundary sets. Every
partition gets an OffCEM estimate and local-correctness DM.
"""
import argparse
from pathlib import Path
import traceback
import warnings

from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from imprecise.cli import add_common_arguments
from imprecise.cli import apply_quick_mode
from imprecise.cli import split_csv
from imprecise.clustering import sample_partitions
from imprecise.experiment import IMPRECISE_METHODS
from imprecise.experiment import evaluate_partition
from imprecise.experiment import fit_method
from imprecise.experiment import load_records
from imprecise.experiment import prepare_seed
from imprecise.experiment import summarize_direction6
from imprecise.experiment import write_json_atomic


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Membership-based partition-sensitivity experiment"
    )
    add_common_arguments(
        parser,
        "/Users/cispa/Documents/OffCEM/imprecise_results/direction6",
    )
    parser.add_argument(
        "--n-partition-samples",
        type=int,
        default=20,
        help="Plausible partitions sampled per uncertain method and seed",
    )
    parser.add_argument(
        "--pcm-reject-threshold",
        type=float,
        default=0.0,
        help=(
            "Keep max-typicality hard labels for PCM actions below this "
            "threshold while sampling the remaining actions"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apply_quick_mode(args, "quick")
    if args.quick:
        args.n_partition_samples = 2
    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    methods = split_csv(args.methods)
    gen_methods = split_csv(args.gen_methods)

    if not args.analyze_only:
        partition_counts = {
            method: (
                1 + args.n_partition_samples
                if method in IMPRECISE_METHODS
                else 1
            )
            for method in methods
        }
        steps_per_seed = (
            1
            + len(methods)
            + sum(partition_counts.values())
        )
        total_steps = len(gen_methods) * args.n_seeds * steps_per_seed
        with tqdm(
            total=total_steps,
            desc="Direction 6",
            unit="step",
            dynamic_ncols=True,
        ) as overall:
            for gen_method in gen_methods:
                for seed_index in range(args.n_seeds):
                    checkpoint = (
                        checkpoint_dir
                        / f"gen-{gen_method}_seed{seed_index:04d}.json"
                    )
                    if checkpoint.exists():
                        overall.update(steps_per_seed)
                        overall.set_postfix_str(
                            f"{gen_method} seed={seed_index} cached"
                        )
                        continue
                    overall.set_postfix_str(
                        f"{gen_method} seed={seed_index} preparing"
                    )
                    prepared = prepare_seed(args, gen_method, seed_index)
                    overall.update()
                    bandit_data = prepared["bandit_data"]
                    record = {
                        "gen_method": gen_method,
                        "seed": seed_index,
                        "true_policy_value": bandit_data["true_policy_value"],
                        "baselines": prepared["baselines"],
                        "methods": [],
                    }
                    method_progress = tqdm(
                        methods,
                        desc=f"{gen_method} seed={seed_index}",
                        unit="method",
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                    )
                    for method in method_progress:
                        method_progress.set_postfix_str(method)
                        expected_partitions = partition_counts[method]
                        clustering_done = False
                        overall.set_postfix_str(
                            f"{gen_method} seed={seed_index} {method}: clustering"
                        )
                        try:
                            clustering = fit_method(
                                args, bandit_data, method, seed_index
                            )
                            overall.update()
                            clustering_done = True
                            partitions = [clustering.labels.copy()]
                            if clustering.scores is not None:
                                partitions.extend(
                                    sample_partitions(
                                        result=clustering,
                                        n_samples=args.n_partition_samples,
                                        random_state=(
                                            args.cluster_seed
                                            + 10000
                                            + seed_index
                                        ),
                                        pcm_reject_threshold=(
                                            args.pcm_reject_threshold
                                        ),
                                    )
                                )
                            method_record = {
                                "method": method,
                                "clustering": {
                                    "uncertainty": clustering.uncertainty,
                                    "metadata": clustering.metadata,
                                },
                                "partitions": [],
                            }
                            draw_progress = tqdm(
                                enumerate(partitions),
                                total=len(partitions),
                                desc=method,
                                unit="partition",
                                leave=False,
                                position=2,
                                dynamic_ncols=True,
                            )
                            for draw, labels in draw_progress:
                                draw_label = f"{method}::draw{draw}"
                                overall.set_postfix_str(
                                    f"{gen_method} seed={seed_index} "
                                    f"{method} draw={draw}"
                                )
                                try:
                                    evaluation = evaluate_partition(
                                        bandit_data=bandit_data,
                                        pi_e=prepared["pi_e"],
                                        q_x_a=prepared["q_x_a"],
                                        labels=labels,
                                        label=draw_label,
                                        model_seed=prepared["model_seed"],
                                    )
                                    estimator = f"OffCEM-2s::{draw_label}"
                                    method_record["partitions"].append(
                                        {
                                            "draw": draw,
                                            "estimate": evaluation[
                                                "estimates"
                                            ][estimator],
                                            "dm_2s": evaluation[
                                                "reward_diagnostics"
                                            ]["dm_2s"],
                                            "dm_1s": evaluation[
                                                "reward_diagnostics"
                                            ]["dm_1s"],
                                            "dm_ratio": evaluation[
                                                "reward_diagnostics"
                                            ]["dm_ratio"],
                                            "ari": evaluation["ari"],
                                            "n_effective_clusters": evaluation[
                                                "n_effective_clusters"
                                            ],
                                        }
                                    )
                                except Exception as error:
                                    method_record["partitions"].append(
                                        {
                                            "draw": draw,
                                            "error": (
                                                f"{type(error).__name__}: "
                                                f"{error}"
                                            ),
                                        }
                                    )
                                    tqdm.write(
                                        f"{draw_label} ERROR: {error}"
                                    )
                                finally:
                                    overall.update()
                            if len(partitions) < expected_partitions:
                                overall.update(
                                    expected_partitions - len(partitions)
                                )
                            record["methods"].append(method_record)
                        except Exception as error:
                            overall.update(
                                expected_partitions
                                + (0 if clustering_done else 1)
                            )
                            record["methods"].append(
                                {
                                    "method": method,
                                    "error": (
                                        f"{type(error).__name__}: {error}"
                                    ),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            tqdm.write(f"{method} ERROR: {error}")
                    write_json_atomic(checkpoint, record)

    records = load_records(output)
    if not records:
        raise SystemExit(f"No checkpoints found in {output}")
    summarize_direction6(records, output)
    errors = [
        (record["gen_method"], record["seed"], item["method"], item["error"])
        for record in records
        for item in record["methods"]
        if "error" in item
    ]
    print(
        f"wrote sensitivity summaries from {len(records)} checkpoints; "
        f"method failures={len(errors)}"
    )
    for error in errors[:10]:
        print("  ", error)


if __name__ == "__main__":
    main()
