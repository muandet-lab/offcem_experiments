"""Compare applicability-runner metrics with existing Figure 7 checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ESTIMATOR_MAP = {
    "IPS": "IPS",
    "DR": "DR",
    "DM": "DM",
    "MIPS-embedding": "MIPS",
    "OffCEM-2s::matched": "OffCEM (true clus + 2s reg)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Figure 7 regression")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    failures = []
    rows = []
    for n_clusters in sorted(metrics["config.dataset.n_clusters"].unique()):
        path = Path(args.reference_dir) / f"results_nc{int(n_clusters)}.json"
        with open(path) as file:
            reference = json.load(file)
        true_value = float(reference["policy_value"])
        for benchmark_name, reference_name in ESTIMATOR_MAP.items():
            selected = metrics.loc[
                (metrics["config.dataset.n_clusters"] == n_clusters)
                & (metrics["estimator"] == benchmark_name)
            ]
            if selected.empty:
                continue
            benchmark_relmse = float(selected.iloc[0]["relMSE"])
            values = np.asarray(reference[reference_name], dtype=float)
            reference_relmse = float(
                np.mean(((values - true_value) / true_value) ** 2)
            )
            relative_error = abs(benchmark_relmse - reference_relmse) / max(
                reference_relmse, 1e-12
            )
            passed = relative_error <= args.relative_tolerance
            rows.append(
                {
                    "n_clusters": int(n_clusters),
                    "estimator": benchmark_name,
                    "benchmark_relMSE": benchmark_relmse,
                    "reference_relMSE": reference_relmse,
                    "relative_error": relative_error,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(rows[-1])
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if failures:
        raise SystemExit(
            f"{len(failures)} Figure 7 comparisons exceeded "
            f"relative tolerance {args.relative_tolerance:.1%}"
        )
    print("Figure 7 regression check passed")


if __name__ == "__main__":
    main()

