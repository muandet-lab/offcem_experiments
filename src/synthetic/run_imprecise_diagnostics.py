"""Direction 1: membership uncertainty as an OffCEM failure diagnostic.

Each method is hardened to one partition. The runner computes OffCEM estimates,
local-correctness DM, and relMSE across paired logged-data seeds.
"""
import argparse
from pathlib import Path
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from imprecise.cli import add_common_arguments
from imprecise.cli import apply_quick_mode
from imprecise.cli import split_csv
from imprecise.experiment import evaluate_partition
from imprecise.experiment import fit_method
from imprecise.experiment import load_records
from imprecise.experiment import prepare_seed
from imprecise.experiment import summarize_direction1
from imprecise.experiment import write_json_atomic


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Membership-uncertainty diagnostic experiment"
    )
    add_common_arguments(
        parser,
        "/Users/cispa/Documents/OffCEM/imprecise_results/direction1",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apply_quick_mode(args, "quick")
    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    methods = split_csv(args.methods)
    gen_methods = split_csv(args.gen_methods)

    if not args.analyze_only:
        steps_per_seed = 1 + 2 * len(methods)
        total_steps = len(gen_methods) * args.n_seeds * steps_per_seed
        with tqdm(
            total=total_steps,
            desc="Direction 1",
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
                            overall.set_postfix_str(
                                f"{gen_method} seed={seed_index} {method}: fitting"
                            )
                            evaluation = evaluate_partition(
                                bandit_data=bandit_data,
                                pi_e=prepared["pi_e"],
                                q_x_a=prepared["q_x_a"],
                                labels=clustering.labels,
                                label=method,
                                model_seed=prepared["model_seed"],
                            )
                            overall.update()
                            record["methods"].append(
                                {
                                    "method": method,
                                    "clustering": {
                                        "uncertainty": clustering.uncertainty,
                                        "metadata": clustering.metadata,
                                        "cluster_sizes": np.bincount(
                                            clustering.labels,
                                            minlength=args.n_clusters,
                                        ).tolist(),
                                    },
                                    "evaluation": evaluation,
                                }
                            )
                        except Exception as error:
                            overall.update(1 if clustering_done else 2)
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
    summarize_direction1(records, output)
    errors = [
        (record["gen_method"], record["seed"], item["method"], item["error"])
        for record in records
        for item in record["methods"]
        if "error" in item
    ]
    print(
        f"wrote summaries from {len(records)} checkpoints; "
        f"method failures={len(errors)}"
    )
    for error in errors[:10]:
        print("  ", error)


if __name__ == "__main__":
    main()
