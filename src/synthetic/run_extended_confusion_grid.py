"""Extend the wfss/dps confusion heatmap to a 5-mechanism grid.

Adds kmeans, spectral, agglomerative as BOTH generating mechanisms and
estimation partitions, alongside the existing wfss (original) / dps
(feature_bucket) rows. Single config, matching the existing compact heatmap:
nc=50, eps=0.2, reward_std=3, n=3000, 10 seeds.

Writes one JSON checkpoint per generating mechanism (a list of per-est-method
row dicts, same schema as run_join_experiment.run_cell), so the existing
plotting code's `load()` (list of rows keyed by est_method) works unchanged.

By default, stochastic estimation partitions are fit with a separate
--cluster-seed. This avoids replaying the reward-generating partition with the
same RNG seed, so "original", "kmeans", etc. mean "same clustering recipe,
without oracle access" rather than an oracle-like reconstruction.

Output layout is flat under --root (default ~/offcem_results), matching the
existing <root>/join_corruption_original and
<root>/join_corruption_feature_bucket checkpoints already produced there:

New rows (kmeans/spectral/agglomerative as generator):
    <root>/join_corruption_<mechanism>/cell_gen-<mechanism>_nc50_eps0.2_rstd3.0_n3000.json
    estimation methods: original, feature_bucket, kmeans, spectral, agglomerative, matched

Extra random-control columns for those new rows are written separately so
existing checkpoints are not overwritten:
    <root>/join_corruption_<mechanism>_random_controls/cell_gen-<mechanism>_..._n3000.json
    estimation methods: random, shuffled_partition

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

import numpy as np

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from ope import run_ope
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy
from run_join_experiment import DEFAULTS
from run_join_experiment import compute_v_true
from run_join_experiment import demeaned_mse
from run_join_experiment import make_dataset
from tqdm import tqdm

NC, EPS, RSTD, N, SEEDS, TEMP = 50, 0.2, 3.0, 3000, 10, 10.0

ALL_METHODS = ["original", "feature_bucket", "kmeans", "spectral", "agglomerative", "matched"]

JOBS = [
    # (out_subdir, gen_method, methods)
    ("join_corruption_kmeans", "kmeans", ALL_METHODS),
    ("join_corruption_spectral", "spectral", ALL_METHODS),
    ("join_corruption_agglomerative", "agglomerative", ALL_METHODS),
    ("join_corruption_kmeans_random_controls", "kmeans", ["random", "shuffled_partition"]),
    ("join_corruption_spectral_random_controls", "spectral", ["random", "shuffled_partition"]),
    ("join_corruption_agglomerative_random_controls", "agglomerative", ["random", "shuffled_partition"]),
    ("join_corruption_original_extra", "original", ["spectral", "agglomerative"]),
    ("join_corruption_feature_bucket_extra", "feature_bucket", ["spectral", "agglomerative"]),
]


def cell_path(out_dir, gen_method):
    return out_dir / f"cell_gen-{gen_method}_nc{NC}_eps{EPS}_rstd{RSTD}_n{N}.json"


def build_estimation_clusterings_decoupled(
    bandit_data,
    methods,
    n_clusters,
    cluster_seed,
):
    """Build estimation partitions without replaying the generator seed."""
    out = []
    for method in methods:
        if method == "matched":
            out.append(
                (
                    "matched",
                    bandit_data["cluster_indices"],
                    bandit_data["clusters"],
                )
            )
            continue

        c1d = compute_clusters(
            bandit_data["action_context_one_hot"],
            n_clusters,
            method=method,
            balance="natural",
            random_state=cluster_seed,
        )
        out.append(
            (
                method,
                c1d,
                clusters_to_onehot_3d(c1d, bandit_data["n_users"]),
            )
        )
    return out


def run_cell_decoupled(
    gen_method,
    methods,
    cluster_seed,
    progress_desc=None,
):
    """One compact heatmap cell with non-oracle estimation seeds."""
    v_true = compute_v_true(gen_method, NC, EPS, RSTD, TEMP)
    per_method = {}
    dataset = make_dataset(RSTD)
    C = DEFAULTS

    with tqdm(
        total=SEEDS * len(methods),
        desc=progress_desc or f"gen={gen_method}",
        unit="fit",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for seed_idx in range(SEEDS):
            model_seed = C["RANDOM_STATE"] + seed_idx
            est_cluster_seed = cluster_seed + seed_idx
            pbar.set_postfix(seed=seed_idx, est="data")
            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=N,
                n_users=C["N_VAL_USERS"],
                n_clusters=NC,
                clustering_method=gen_method,
                cluster_balance="natural",
                cluster_temperature=TEMP,
            )
            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"],
                eps=EPS,
            )
            expected_reward = bandit_data["expected_reward"]

            for label, c1d, c3d in build_estimation_clusterings_decoupled(
                bandit_data,
                methods,
                NC,
                est_cluster_seed,
            ):
                pbar.set_postfix(seed=seed_idx, est=label)
                f_x_a, q_x_a = train_reward_model_via_two_stage(
                    bandit_data,
                    c3d,
                    random_state=model_seed,
                )
                est_vals = run_ope(
                    bandit_data=bandit_data,
                    pi_e=pi_e,
                    action_clusters=c3d,
                    f_x_a=f_x_a,
                    q_x_a=q_x_a,
                )
                dm2 = demeaned_mse(f_x_a[:, :, 0], expected_reward, c1d)
                dm1 = demeaned_mse(q_x_a[:, :, 0], expected_reward, c1d)
                acc = per_method.setdefault(
                    label,
                    dict(est_vals=[], dm_2s=[], dm_1s=[]),
                )
                acc["est_vals"].append(est_vals)
                acc["dm_2s"].append(dm2)
                acc["dm_1s"].append(dm1)
                pbar.update(1)

    rows = []
    norm = v_true ** 2
    for label, acc in per_method.items():
        metrics = {}
        for key in acc["est_vals"][0].keys():
            values = np.array([d[key] for d in acc["est_vals"]])
            bias = float(values.mean() - v_true)
            var = float(values.var(ddof=0))
            metrics[key] = dict(
                relMSE=(bias ** 2 + var) / norm,
                relBias2=bias ** 2 / norm,
                relVar=var / norm,
                est_mean=float(values.mean()),
            )
        dm2 = np.array(acc["dm_2s"])
        dm1 = np.array(acc["dm_1s"])
        rows.append(
            dict(
                gen_method=gen_method,
                n_clusters=NC,
                eps=EPS,
                reward_std=RSTD,
                n=N,
                est_method=label,
                n_seeds=SEEDS,
                v_true=v_true,
                dm_2s_mean=float(dm2.mean()),
                dm_2s_std=float(dm2.std()),
                dm_1s_mean=float(dm1.mean()),
                dm_ratio_mean=float((dm2 / dm1).mean()),
                metrics=metrics,
                cluster_seed=cluster_seed,
            )
        )
    return rows


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
    p.add_argument(
        "--cluster-seed",
        type=int,
        default=22345,
        help="Seed for estimation clustering. Keep different from 12345 to avoid "
        "oracle-like replay of stochastic reward-generating partitions.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSON checkpoints instead of skipping them.",
    )
    args = p.parse_args()
    root = Path(args.root)

    for subdir, gen_method, methods in JOBS:
        out_dir = root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = cell_path(out_dir, gen_method)
        if ckpt.exists() and not args.force:
            print(f"[skip] {ckpt} already exists")
            continue
        if ckpt.exists() and args.force:
            print(f"[force] overwriting {ckpt}")
        print(
            f"[run] gen={gen_method} methods={methods} "
            f"cluster_seed={args.cluster_seed} -> {ckpt}"
        )
        t0 = time()
        rows = run_cell_decoupled(
            gen_method,
            methods,
            args.cluster_seed,
            progress_desc=f"gen={gen_method}",
        )
        with open(ckpt, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[done] {ckpt} ({len(rows)} rows) in {(time()-t0)/60:.1f} min")
    print("ALL JOBS COMPLETE")


if __name__ == "__main__":
    main()
