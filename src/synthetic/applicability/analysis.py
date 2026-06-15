"""Aggregate benchmark checkpoints and classify paired estimator comparisons."""
import json
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


def analyze_results(
    output_dir: str,
    formal_inference: bool,
    expected_checkpoints=None,
    bootstrap_samples: int = 5000,
    alpha: float = 0.05,
    minimum_relative_effect: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = _load_records(
        Path(output_dir),
        expected_checkpoints=expected_checkpoints,
    )
    if not records:
        raise ValueError(f"No successful checkpoints found in {output_dir}")

    metric_rows = []
    comparison_rows = []
    grouped = {}
    for record in records:
        for partition in record["partitions"]:
            if partition.get("error"):
                continue
            key = (record["cell_id"], partition["label"])
            grouped.setdefault(key, []).append((record, partition))

    for (cell_id, partition_label), group in sorted(grouped.items()):
        first_record, first_partition = group[0]
        estimator_values = {}
        true_values = []
        for record, partition in group:
            true_values.append(float(record["true_policy_value"]))
            estimates = dict(record["shared_estimates"])
            estimates.update(partition["estimates"])
            for estimator, value in estimates.items():
                estimator_values.setdefault(estimator, []).append(float(value))

        true_values = np.asarray(true_values)
        for estimator, values in estimator_values.items():
            values = np.asarray(values)
            metric_rows.append(
                {
                    "cell_id": cell_id,
                    "partition": partition_label,
                    "estimator": estimator,
                    "n_seeds": len(values),
                    **_flatten_scalars(first_record["cell"], prefix="config"),
                    **_flatten_scalars(
                        first_record.get("nominal_variance_shares", {}),
                        prefix="nominal_share",
                    ),
                    **_flatten_scalars(
                        first_record.get("realized_variance_shares", {}),
                        prefix="realized_share",
                    ),
                    **_estimator_metrics(values, true_values),
                }
            )

        offcem_name = f"OffCEM-2s::{partition_label}"
        offcem = np.asarray(estimator_values[offcem_name])
        offcem_error = ((offcem - true_values) / true_values) ** 2
        for baseline in ("DR", "DM", "MIPS-embedding", "IPS", "SNIPS"):
            if baseline not in estimator_values:
                continue
            baseline_values = np.asarray(estimator_values[baseline])
            baseline_error = ((baseline_values - true_values) / true_values) ** 2
            differences = offcem_error - baseline_error
            ci_low, ci_high, p_value = _paired_bootstrap(
                differences,
                n_bootstrap=bootstrap_samples,
                seed=_stable_seed(cell_id, partition_label, baseline),
            )
            comparison_rows.append(
                {
                    "cell_id": cell_id,
                    "partition": partition_label,
                    "baseline": baseline,
                    "n_seeds": differences.size,
                    "offcem_relMSE": float(offcem_error.mean()),
                    "baseline_relMSE": float(baseline_error.mean()),
                    "paired_delta_relMSE": float(differences.mean()),
                    "relative_effect": float(
                        differences.mean() / max(baseline_error.mean(), 1e-12)
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "phase": first_record.get("phase", ""),
                    **_flatten_scalars(first_record["cell"], prefix="config"),
                    **_flatten_scalars(
                        first_record.get("nominal_variance_shares", {}),
                        prefix="nominal_share",
                    ),
                    **_flatten_scalars(
                        first_record.get("realized_variance_shares", {}),
                        prefix="realized_share",
                    ),
                    **_flatten_diagnostics(first_partition.get("diagnostics", {})),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["p_adjusted"] = np.nan
    comparisons["classification"] = "screen-only"
    if formal_inference and not comparisons.empty:
        for baseline in comparisons["baseline"].unique():
            mask = comparisons["baseline"] == baseline
            comparisons.loc[mask, "p_adjusted"] = _benjamini_hochberg(
                comparisons.loc[mask, "p_value"].to_numpy()
            )
        comparisons["classification"] = comparisons.apply(
            lambda row: _classify(
                row,
                alpha=alpha,
                minimum_relative_effect=minimum_relative_effect,
            ),
            axis=1,
        )

    metrics.to_csv(Path(output_dir) / "estimator_metrics.csv", index=False)
    comparisons.to_csv(Path(output_dir) / "paired_comparisons.csv", index=False)
    return metrics, comparisons


def _load_records(
    output: Path,
    expected_checkpoints=None,
) -> List[Dict]:
    records = []
    for path in sorted((output / "checkpoints").glob("*.json")):
        if (
            expected_checkpoints is not None
            and path.name not in expected_checkpoints
        ):
            continue
        with open(path) as file:
            record = json.load(file)
        if record.get("status") == "ok":
            records.append(record)
    return records


def _estimator_metrics(values: np.ndarray, true_values: np.ndarray) -> Dict:
    normalized_errors = (values - true_values) / true_values
    return {
        "estimate_mean": float(values.mean()),
        "true_value_mean": float(true_values.mean()),
        "relMSE": float(np.mean(normalized_errors**2)),
        "relBias2": float(np.mean(normalized_errors) ** 2),
        "relVar": float(np.var(normalized_errors)),
    }


def _paired_bootstrap(
    differences: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    if differences.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    indices = rng.randint(
        0, differences.size, size=(n_bootstrap, differences.size)
    )
    means = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])

    centered = differences - differences.mean()
    null_means = centered[
        rng.randint(0, differences.size, size=(n_bootstrap, differences.size))
    ].mean(axis=1)
    observed = abs(differences.mean())
    p_value = (np.count_nonzero(np.abs(null_means) >= observed) + 1) / (
        n_bootstrap + 1
    )
    return float(ci_low), float(ci_high), float(p_value)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    values = p_values[finite]
    if values.size == 0:
        return adjusted
    order = np.argsort(values)
    ranked = values[order]
    corrected = ranked * values.size / np.arange(1, values.size + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.minimum(corrected, 1.0)
    adjusted[finite] = restored
    return adjusted


def _classify(
    row: pd.Series,
    alpha: float,
    minimum_relative_effect: float,
) -> str:
    if not np.isfinite(row["p_adjusted"]) or row["p_adjusted"] > alpha:
        return "tie"
    if (
        row["ci_high"] < 0
        and row["relative_effect"] <= -minimum_relative_effect
    ):
        return "win"
    if (
        row["ci_low"] > 0
        and row["relative_effect"] >= minimum_relative_effect
    ):
        return "loss"
    return "tie"


def _flatten_diagnostics(diagnostics: Dict) -> Dict:
    flattened = {}
    for group, values in diagnostics.items():
        if isinstance(values, dict):
            for name, value in values.items():
                flattened[f"{group}.{name}"] = value
    return flattened


def _flatten_scalars(values: Dict, prefix: str) -> Dict:
    flattened = {}
    for name, value in values.items():
        key = f"{prefix}.{name}"
        if isinstance(value, dict):
            flattened.update(_flatten_scalars(value, key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
    return flattened


def _stable_seed(*parts: Iterable[str]) -> int:
    text = "::".join(map(str, parts))
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (
        2**31 - 1
    )
