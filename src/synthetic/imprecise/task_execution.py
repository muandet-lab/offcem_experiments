"""Common task-array helpers for checkpointed experiment runners."""
import json
import os


TASK_INDEX_ENV_VAR = "OFFCEM_TASK_INDEX"


def add_task_arguments(parser):
    parser.add_argument(
        "--task-index",
        type=int,
        help="Run one zero-based experiment cell without aggregation",
    )
    parser.add_argument(
        "--print-task-count",
        action="store_true",
        help="Print the number of independent experiment cells and exit",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Report checkpoint completion without running or aggregating",
    )
    parser.add_argument(
        "--no-task-env",
        action="store_true",
        help=f"Ignore {TASK_INDEX_ENV_VAR}",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow analysis with missing or failed checkpoints",
    )


def validate_task_arguments(args):
    if args.analyze_only and args.status_only:
        raise SystemExit("--analyze-only and --status-only are exclusive")
    if args.task_index is not None and (
        args.analyze_only or args.status_only
    ):
        raise SystemExit(
            "--task-index cannot be combined with analysis or status modes"
        )


def selected_task_index(args, n_tasks):
    if args.print_task_count or args.analyze_only or args.status_only:
        return None
    value = args.task_index
    if value is None and not args.no_task_env:
        environment_value = os.environ.get(TASK_INDEX_ENV_VAR)
        if environment_value is not None:
            value = int(environment_value)
            print(f"using {TASK_INDEX_ENV_VAR}={value}")
    if value is not None and not 0 <= value < n_tasks:
        raise SystemExit(
            f"task index {value} is outside the valid range "
            f"0..{n_tasks - 1}"
        )
    return value


def checkpoint_summary(checkpoint_paths, failure_count):
    summary = {
        "expected": len(checkpoint_paths),
        "ok": 0,
        "error": 0,
        "missing": 0,
        "details": {"error": [], "missing": []},
    }
    for path in checkpoint_paths:
        if not path.exists():
            summary["missing"] += 1
            summary["details"]["missing"].append(path.name)
            continue
        try:
            with open(path) as file:
                record = json.load(file)
            failures = failure_count(record)
        except Exception as error:
            failures = 1
            summary["details"]["error"].append(
                f"{path.name}: {type(error).__name__}: {error}"
            )
        if failures:
            summary["error"] += 1
            if not summary["details"]["error"] or not summary[
                "details"
            ]["error"][-1].startswith(path.name):
                summary["details"]["error"].append(
                    f"{path.name}: {failures} recorded failure(s)"
                )
        else:
            summary["ok"] += 1
    return summary


def print_checkpoint_summary(summary):
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
            print(f"  ... {remaining} more {category} checkpoints")
