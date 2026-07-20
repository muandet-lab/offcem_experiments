"""Extend the wfss/dps confusion heatmap to a 5-mechanism grid.

Adds kmeans, spectral, agglomerative as BOTH generating mechanisms and
estimation partitions, alongside the existing wfss (original) / dps
(feature_bucket) rows. Single config, matching the existing compact heatmap:
nc=50, eps=0.2, reward_std=3, n=3000, 10 seeds.

Writes one JSON checkpoint per generating mechanism (a list of per-est-method
row dicts, same schema as run_join_experiment.run_cell), so the existing
plotting code's `load()` (list of rows keyed by est_method) works unchanged.

Output layout is flat under --root (default ~/offcem_results), matching the
existing <root>/join_corruption_original and
<root>/join_corruption_feature_bucket checkpoints already produced there:

New rows (kmeans/spectral/agglomerative as generator):
    <root>/join_corruption_<mechanism>/cell_gen-<mechanism>_nc50_eps0.2_rstd3.0_n3000.json
    estimation methods: original, feature_bucket, kmeans, spectral, agglomerative, matched

Extra columns for existing rows (wfss/dps as generator, spectral/agglomerative
as new estimation columns -- kept in a separate "_extra" dir so the original
checkpoints are never touched or reinterpreted):
    <root>/join_corruption_original_extra/cell_gen-original_..._n3000.json
    <root>/join_corruption_feature_bucket_extra/cell_gen-feature_bucket_..._n3000.json
    estimation methods: spectral, agglomerative
"""
import argparse
import json
import warnings
from pathlib import Path
from time import time

warnings.filterwarnings("ignore")

from run_join_experiment import run_cell

NC, EPS, RSTD, N, SEEDS, TEMP = 50, 0.2, 3.0, 3000, 10, 10.0

ALL_METHODS = ["original", "feature_bucket", "kmeans", "spectral", "agglomerative", "matched"]

JOBS = [
    # (out_subdir, gen_method, methods)
    ("join_corruption_kmeans", "kmeans", ALL_METHODS),
    ("join_corruption_spectral", "spectral", ALL_METHODS),
    ("join_corruption_agglomerative", "agglomerative", ALL_METHODS),
    ("join_corruption_original_extra", "original", ["spectral", "agglomerative"]),
    ("join_corruption_feature_bucket_extra", "feature_bucket", ["spectral", "agglomerative"]),
]


def cell_path(out_dir, gen_method):
    return out_dir / f"cell_gen-{gen_method}_nc{NC}_eps{EPS}_rstd{RSTD}_n{N}.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=str(Path.home() / "offcem_results"),
        help="Base output directory. Subdirectories are created flat inside it, "
        "e.g. <root>/join_corruption_kmeans/cell_gen-kmeans_....json -- matching "
        "the existing <root>/join_corruption_original and "
        "<root>/join_corruption_feature_bucket layout.",
    )
    args = p.parse_args()
    root = Path(args.root)

    for subdir, gen_method, methods in JOBS:
        out_dir = root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = cell_path(out_dir, gen_method)
        if ckpt.exists():
            print(f"[skip] {ckpt} already exists")
            continue
        print(f"[run] gen={gen_method} methods={methods} -> {ckpt}")
        t0 = time()
        rows = run_cell(
            gen_method, NC, EPS, RSTD, N, methods, [], SEEDS, TEMP,
            progress_desc=f"gen={gen_method}",
        )
        with open(ckpt, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[done] {ckpt} ({len(rows)} rows) in {(time()-t0)/60:.1f} min")
    print("ALL JOBS COMPLETE")


if __name__ == "__main__":
    main()
