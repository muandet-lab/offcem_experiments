"""Run the isolated OffCEM applicability benchmark.

Examples:
    conda run -n offcem python -m run_applicability_benchmark \
        --config applicability/configs/smoke.yaml --dry-run

    conda run -n offcem python -m run_applicability_benchmark \
        --config applicability/configs/smoke.yaml
"""
import argparse
import os
from pathlib import Path

from applicability.analysis import analyze_results
from applicability.config import build_cells
from applicability.config import checkpoint_name
from applicability.config import estimate_runtime_seconds
from applicability.config import load_config
from applicability.config import task_count
from applicability.config import task_index
from applicability.config import write_manifest
from applicability.runner import checkpoint_summary
from applicability.runner import run_benchmark
from applicability.runner import run_task_index


TASK_INDEX_ENV_VARS = ("OFFCEM_TASK_INDEX",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OffCEM applicability benchmark"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        help="Override execution.output_dir (or set OFFCEM_OUTPUT_DIR)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-task-count", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        "--analyze-only",
        dest="aggregate_only",
        action="store_true",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument(
        "--no-task-env",
        action="store_true",
        help="Ignore scheduler task-index environment variables",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow aggregation with missing or failed checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    cells = build_cells(config)
    execution = config.get("execution", {})
    n_seeds = int(execution.get("n_seeds", 5))
    if args.print_task_count:
        print(task_count(cells, n_seeds))
        return

    output_value = (
        args.output_dir
        or os.environ.get("OFFCEM_OUTPUT_DIR")
        or execution["output_dir"]
    )
    output_dir = Path(output_value)
    selected_task = _selected_task_index(args, n_seeds)
    if selected_task is None:
        write_manifest(
            args.config,
            config,
            cells,
            output_dir,
            n_seeds=n_seeds,
        )

    estimated_seconds = estimate_runtime_seconds(config, cells)
    print(
        f"phase={execution.get('phase', 'screening')} "
        f"cells={len(cells)} seeds={n_seeds} "
        f"dataset-runs={len(cells) * n_seeds}"
    )
    if estimated_seconds:
        print(
            f"estimated runtime={estimated_seconds / 3600:.2f}h "
            "(calibrate runtime.reference_seconds from Phase A)"
        )
    else:
        print("estimated runtime=unknown (run Phase A timing first)")
    print(f"output={output_dir}")
    if args.prepare_only:
        print(f"wrote {output_dir / 'manifest.json'}")
        print(f"wrote {output_dir / 'tasks.tsv'}")
        return

    if args.dry_run:
        if selected_task is not None:
            cell_index = selected_task // n_seeds
            seed_index = selected_task % n_seeds
            print(
                f"selected task={selected_task} cell={cell_index} "
                f"seed={seed_index}"
            )
        return

    if args.status_only:
        _print_checkpoint_summary(
            checkpoint_summary(cells, str(output_dir), n_seeds)
        )
        return

    if selected_task is not None and not args.aggregate_only:
        result = run_task_index(
            cells=cells,
            output_dir=str(output_dir),
            n_seeds=n_seeds,
            base_sample_seed=int(execution.get("base_sample_seed", 50000)),
            phase=execution.get("phase", "screening"),
            task_index=selected_task,
            force=args.force,
        )
        if result["status"] != "ok":
            raise SystemExit(1)
        return

    if not args.aggregate_only:
        run_benchmark(
            cells=cells,
            output_dir=str(output_dir),
            n_seeds=n_seeds,
            base_sample_seed=int(execution.get("base_sample_seed", 50000)),
            phase=execution.get("phase", "screening"),
            force=args.force,
        )

    summary = checkpoint_summary(cells, str(output_dir), n_seeds)
    _print_checkpoint_summary(summary)
    if (
        not args.allow_incomplete
        and (summary["missing"] > 0 or summary["error"] > 0)
    ):
        raise SystemExit(
            "Refusing to aggregate incomplete results. "
            "Retry failed/missing array tasks or pass --allow-incomplete."
        )

    metrics, comparisons = analyze_results(
        output_dir=str(output_dir),
        expected_checkpoints={
            f"{task['checkpoint']}"
            for task in _expected_tasks(cells, n_seeds)
        },
        formal_inference=bool(execution.get("formal_inference", False)),
        bootstrap_samples=int(execution.get("bootstrap_samples", 5000)),
        alpha=float(execution.get("alpha", 0.05)),
        minimum_relative_effect=float(
            execution.get("minimum_relative_effect", 0.10)
        ),
    )
    print(
        f"wrote {len(metrics)} estimator metrics and "
        f"{len(comparisons)} paired comparisons"
    )


def _selected_task_index(args, n_seeds):
    if args.task_index is not None and (
        args.cell_index is not None or args.seed_index is not None
    ):
        raise SystemExit(
            "--task-index cannot be combined with --cell-index/--seed-index"
        )
    if (args.cell_index is None) != (args.seed_index is None):
        raise SystemExit("--cell-index and --seed-index must be provided together")
    if args.task_index is not None:
        return args.task_index
    if args.cell_index is not None:
        return task_index(args.cell_index, args.seed_index, n_seeds)
    if args.aggregate_only or args.prepare_only or args.status_only:
        return None
    if not args.no_task_env:
        for name in TASK_INDEX_ENV_VARS:
            value = os.environ.get(name)
            if value is not None:
                print(f"using {name}={value}")
                return int(value)
    return None


def _validate_args(args):
    modes = sum(
        bool(value)
        for value in (
            args.prepare_only,
            args.aggregate_only,
            args.status_only,
        )
    )
    if modes > 1:
        raise SystemExit(
            "--prepare-only, --aggregate-only, and --status-only are exclusive"
        )
    has_task_selection = (
        args.task_index is not None
        or args.cell_index is not None
        or args.seed_index is not None
    )
    if has_task_selection and modes:
        raise SystemExit(
            "Task selection cannot be combined with prepare/status/aggregate modes"
        )


def _expected_tasks(cells, n_seeds):
    for cell in cells:
        for seed_index in range(n_seeds):
            yield {
                "checkpoint": checkpoint_name(cell, seed_index),
            }


def _print_checkpoint_summary(summary):
    print(
        "checkpoints: "
        f"expected={summary['expected']} ok={summary['ok']} "
        f"error={summary['error']} missing={summary['missing']}"
    )
    for category in ("error", "missing"):
        for item in summary["details"][category][:10]:
            print(f"  {category}: {item}")
        remaining = len(summary["details"][category]) - 10
        if remaining > 0:
            print(f"  ... {remaining} more {category} tasks")


if __name__ == "__main__":
    main()
