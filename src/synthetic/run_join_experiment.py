"""Unified JOIN experiment: does the within-cluster demeaned residual (dm, the
local-correctness violation, Objective 2) predict OffCEM's estimator error
(relMSE / relBias2, Objective 1)?

This is the *decisive* check the other findings deferred. It is deliberately
built to NOT be a single manufactured knob (unlike the standalone corrupt_p
sweep, where dm and relMSE co-move trivially through one common cause). Instead
it pairs both quantities on identical runs and varies dm through several
INDEPENDENT mechanisms, plus DECOUPLING knobs that would break a naive dm->relMSE
link if dm were not the right statistic:

  Mechanisms that move dm (different routes):
    --methods           estimation clustering (matched/kmeans/random/.../corrupt)
    --n-clusters        cluster granularity
    --corrupt-p-list    graded corruption continuum (for the 'corrupt' method)
    --gen-clustering-method   reward feature-coherence (original=arbitrary,
                              feature_bucket=feature-coherent)

  Decoupling knobs (adversarial to a naive proxy):
    --eps-list          target policy -> changes [pi_e - w*pi_b], i.e. the bias
                        weighting FOR FIXED dm. If dm still predicts bias across
                        eps, strong; if the slope shifts, the policy-weighted
                        form is needed (theory predicts this).
    --n-list            logged sample size -> moves VARIANCE only (dm fixed).
    --reward-std-list   reward noise -> moves variance (and some dm).

Each (cell x estimation-method) row stores, paired on the SAME data:
  - dm_2s / dm_1s          within-cluster demeaned MSE (local-correctness violation)
  - relMSE / relBias2 / relVar  for OffCEM-2s and OffCEM-1s (vs V_true)
This lets the analysis (`--analyze-only`) test whether relBias2 collapses onto a
single function of dm ACROSS mechanisms (proxy valid) or splits by mechanism
(proxy insufficient) -- the pre-registered decision rule for "Story B".

Theory note: bias = sum_c sum_{a in c} [pi_b*w - pi_e] (delta - delta_bar_c),
so dm relates to BIAS, not directly to relMSE (= bias^2 + variance). Expect
relBias2-vs-dm to be the clean relationship and relMSE-vs-dm to carry extra
variance scatter from --n-list/--n-clusters. That is expected, not a failure.

Usage:
    cd /Users/cispa/Documents/OffCEM/icml2023-offcem-expts/src/synthetic
    python -m run_join_experiment --quick            # tiny smoke
    python -m run_join_experiment \
        --methods matched,kmeans,random,shuffled_partition,corrupt \
        --n-clusters 10,50,200 --eps-list 0.0,0.2,0.7 \
        --n-list 1000,3000 --reward-std-list 1,3 \
        --corrupt-p-list 0.25,0.75 --n-seeds 10
    python -m run_join_experiment --analyze-only      # rebuild plots from checkpoints
"""
import argparse
import itertools
import json
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from time import time

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from clustering import corrupt_partition
from dataset import SyntheticBanditDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import linear_reward_function
from ope import run_ope
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tqdm import tqdm

DEFAULTS = dict(
    DIM_CONTEXT=10,
    N_VAL_USERS=200,
    N_TEST_DATA=100000,
    N_TEST_USERS=1000,
    N_ACTIONS=1000,
    N_CAT_PER_DIM=5,
    LATENT_PARAM_MAT_DIM=5,
    N_CAT_DIM=10,
    BETA=-0.1,
    N_DEF_ACTIONS=0.0,
    REWARD_TYPE="continuous",
    RANDOM_STATE=12345,
)

OFFCEM_2S = "OffCEM (true clus + 2s reg)"
OFFCEM_1S = "OffCEM (true clus + 1s reg)"


def parse_args():
    p = argparse.ArgumentParser(description="Unified dm<->relMSE join experiment")
    p.add_argument("--methods", type=str, default="matched,kmeans,random,corrupt")
    p.add_argument("--n-clusters", type=str, default="10,50,200")
    p.add_argument("--eps-list", type=str, default="0.0,0.2,0.7")
    p.add_argument("--n-list", type=str, default="1000,3000")
    p.add_argument("--reward-std-list", type=str, default="1,3")
    p.add_argument("--corrupt-p-list", type=str, default="0.25,0.75")
    p.add_argument(
        "--gen-clustering-method",
        type=str,
        default="original",
        help="Reward-generating partition. 'original'=feature-arbitrary reward; "
        "'feature_bucket'=feature-coherent reward (escapes the adversarial-benchmark critique).",
    )
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/join_results",
    )
    p.add_argument("--quick", action="store_true", help="tiny smoke: 2 seeds, small grid")
    p.add_argument("--analyze-only", action="store_true", help="only (re)build plots from checkpoints")
    return p.parse_args()


def make_dataset(reward_std):
    C = DEFAULTS
    return SyntheticBanditDataset(
        n_actions=C["N_ACTIONS"],
        dim_context=C["DIM_CONTEXT"],
        beta=C["BETA"],
        reward_type=C["REWARD_TYPE"],
        n_cat_per_dim=C["N_CAT_PER_DIM"],
        latent_param_mat_dim=C["LATENT_PARAM_MAT_DIM"],
        n_cat_dim=C["N_CAT_DIM"],
        n_deficient_actions=int(C["N_ACTIONS"] * C["N_DEF_ACTIONS"]),
        reward_function=linear_reward_function,
        reward_std=reward_std,
        random_state=C["RANDOM_STATE"],
    )


def compute_v_true(gen_method, n_clusters, eps, reward_std, temperature):
    """Ground-truth policy value of the eps-greedy target on a large test set.

    Computed on a FRESH dataset instance so it is deterministic per
    (gen_method, n_clusters, eps) and does not perturb the seed-loop RNG.
    """
    C = DEFAULTS
    ds = make_dataset(reward_std)
    test_data = ds.obtain_batch_bandit_feedback(
        n_rounds=C["N_TEST_DATA"],
        n_users=C["N_TEST_USERS"],
        n_clusters=n_clusters,
        clustering_method=gen_method,
        cluster_balance="natural",
        cluster_temperature=temperature,
    )
    return float(
        np.average(
            test_data["expected_reward"],
            weights=gen_eps_greedy(expected_reward=test_data["expected_reward"], eps=eps)[:, :, 0],
            axis=1,
        ).mean()
    )


def demeaned_mse(pred_2d, expected_reward, clusters_1d):
    """Within-cluster demeaned MSE -- the L2 magnitude of the local-correctness
    violation (Assumption 3.1). Mirrors run_local_correctness.analyze_local_correctness.
    """
    err = pred_2d - expected_reward
    err_dm = err.copy()
    for c in range(int(clusters_1d.max()) + 1):
        mask = clusters_1d == c
        if mask.any():
            err_dm[:, mask] -= err[:, mask].mean(axis=1, keepdims=True)
    return float((err_dm ** 2).mean())


def build_estimation_clusterings(bandit_data, methods, n_clusters, corrupt_p_list, seed):
    """Return list of (label, clusters_1d, clusters_3d). Estimation partitions use
    a FIXED random_state across seeds, so cross-seed variance is pure data sampling.
    """
    out = []
    for m in methods:
        if m == "matched":
            out.append(("matched", bandit_data["cluster_indices"], bandit_data["clusters"]))
        elif m == "corrupt":
            for p in corrupt_p_list:
                c1d = corrupt_partition(bandit_data["cluster_indices"], p, seed)
                out.append((f"corrupt_p{p}", c1d, clusters_to_onehot_3d(c1d, bandit_data["n_users"])))
        else:
            c1d = compute_clusters(
                bandit_data["action_context_one_hot"], n_clusters,
                method=m, balance="natural", random_state=seed,
            )
            out.append((m, c1d, clusters_to_onehot_3d(c1d, bandit_data["n_users"])))
    return out


def run_cell(gen_method, n_clusters, eps, reward_std, n, methods, corrupt_p_list, n_seeds, temperature):
    """One factor cell: returns one row per estimation method, paired (dm, relMSE)."""
    C = DEFAULTS
    v_true = compute_v_true(gen_method, n_clusters, eps, reward_std, temperature)

    # per estimation-method accumulators
    per_method = {}  # label -> dict(est_vals=list of dicts, dm_2s=[], dm_1s=[])

    dataset = make_dataset(reward_std)
    for seed_idx in range(n_seeds):
        seed = C["RANDOM_STATE"] + seed_idx
        bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=n,
            n_users=C["N_VAL_USERS"],
            n_clusters=n_clusters,
            clustering_method=gen_method,
            cluster_balance="natural",
            cluster_temperature=temperature,
        )
        pi_e = gen_eps_greedy(expected_reward=bandit_data["expected_reward"], eps=eps)
        expected_reward = bandit_data["expected_reward"]

        for label, c1d, c3d in build_estimation_clusterings(
            bandit_data, methods, n_clusters, corrupt_p_list, C["RANDOM_STATE"]
        ):
            f_x_a, q_x_a = train_reward_model_via_two_stage(bandit_data, c3d, random_state=seed)
            est_vals = run_ope(
                bandit_data=bandit_data, pi_e=pi_e, action_clusters=c3d,
                f_x_a=f_x_a, q_x_a=q_x_a,
            )
            dm2 = demeaned_mse(f_x_a[:, :, 0], expected_reward, c1d)
            dm1 = demeaned_mse(q_x_a[:, :, 0], expected_reward, c1d)
            acc = per_method.setdefault(label, dict(est_vals=[], dm_2s=[], dm_1s=[]))
            acc["est_vals"].append(est_vals)
            acc["dm_2s"].append(dm2)
            acc["dm_1s"].append(dm1)

    rows = []
    norm = v_true ** 2
    for label, acc in per_method.items():
        metrics = {}
        for key in acc["est_vals"][0].keys():
            v = np.array([d[key] for d in acc["est_vals"]])
            bias = float(v.mean() - v_true)
            var = float(v.var(ddof=0))
            metrics[key] = dict(
                relMSE=(bias ** 2 + var) / norm,
                relBias2=bias ** 2 / norm,
                relVar=var / norm,
                est_mean=float(v.mean()),
            )
        dm2 = np.array(acc["dm_2s"])
        dm1 = np.array(acc["dm_1s"])
        rows.append(dict(
            gen_method=gen_method, n_clusters=n_clusters, eps=eps,
            reward_std=reward_std, n=n, est_method=label, n_seeds=n_seeds,
            v_true=v_true,
            dm_2s_mean=float(dm2.mean()), dm_2s_std=float(dm2.std()),
            dm_1s_mean=float(dm1.mean()),
            dm_ratio_mean=float((dm2 / dm1).mean()),
            metrics=metrics,
        ))
    return rows


def cell_key(gen_method, n_clusters, eps, reward_std, n):
    return f"cell_gen-{gen_method}_nc{n_clusters}_eps{eps}_rstd{reward_std}_n{n}.json"


def run_grid(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods.split(",")
    nc_list = [int(x) for x in args.n_clusters.split(",")]
    eps_list = [float(x) for x in args.eps_list.split(",")]
    n_list = [int(x) for x in args.n_list.split(",")]
    rstd_list = [float(x) for x in args.reward_std_list.split(",")]
    corrupt_p_list = [float(x) for x in args.corrupt_p_list.split(",")] if args.corrupt_p_list else []

    cells = list(itertools.product(nc_list, eps_list, rstd_list, n_list))
    print(f"{len(cells)} cells x {len(methods)} methods x {args.n_seeds} seeds "
          f"(gen={args.gen_clustering_method})")

    for i, (nc, eps, rstd, n) in enumerate(cells, 1):
        ckpt = out_dir / cell_key(args.gen_clustering_method, nc, eps, rstd, n)
        if ckpt.exists():
            print(f"[{i}/{len(cells)}] {ckpt.name}: cached")
            continue
        print(f"[{i}/{len(cells)}] nc={nc} eps={eps} rstd={rstd} n={n} ...")
        t0 = time()
        rows = run_cell(args.gen_clustering_method, nc, eps, rstd, n,
                        methods, corrupt_p_list, args.n_seeds, args.temperature)
        with open(ckpt, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"    -> {len(rows)} rows in {(time()-t0)/60:.1f} min")


# ── Analysis ──


def load_rows(out_dir):
    rows = []
    for f in sorted(Path(out_dir).glob("cell_*.json")):
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def _safe_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    sr = spearmanr(x[ok], y[ok]).correlation
    pr = pearsonr(np.log10(x[ok] + 1e-12), np.log10(y[ok] + 1e-12))[0]
    return float(sr), float(pr)


def analyze(out_dir):
    rows = load_rows(out_dir)
    if not rows:
        print("No checkpoints to analyze.")
        return
    dm = np.array([r["dm_2s_mean"] for r in rows])
    relbias2 = np.array([r["metrics"].get(OFFCEM_2S, {}).get("relBias2", np.nan) for r in rows])
    relmse = np.array([r["metrics"].get(OFFCEM_2S, {}).get("relMSE", np.nan) for r in rows])
    relvar = np.array([r["metrics"].get(OFFCEM_2S, {}).get("relVar", np.nan) for r in rows])
    methods = [r["est_method"] for r in rows]
    eps = np.array([r["eps"] for r in rows])

    print(f"\n=== JOIN analysis ({len(rows)} cell x method points) ===")
    print("Pre-registered decision: relBias2 should collapse onto a single")
    print("function of dm across mechanisms (proxy valid); relMSE may carry")
    print("extra variance scatter (expected). relVar vs dm should be ~uncorrelated.\n")
    for name, y in [("relBias2", relbias2), ("relMSE", relmse), ("relVar", relvar)]:
        sr, pr = _safe_corr(dm, y)
        print(f"  dm_2s vs {name:9s}: Spearman={sr:+.3f}  Pearson(log-log)={pr:+.3f}")
    print("\n  relBias2 vs dm WITHIN each eps (tests policy-weight decoupling):")
    for e in sorted(set(eps.tolist())):
        m = eps == e
        sr, _ = _safe_corr(dm[m], relbias2[m])
        print(f"    eps={e}: Spearman={sr:+.3f}  (n={int(m.sum())})")

    # scatter plots
    uniq = sorted(set(methods))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
    cby = {mm: cmap[i] for i, mm in enumerate(uniq)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, y, title in [
        (axes[0], relbias2, "relBias2 vs dm  (should COLLAPSE if proxy valid)"),
        (axes[1], relmse, "relMSE vs dm  (variance adds scatter)"),
        (axes[2], relvar, "relVar vs dm  (should be ~flat)"),
    ]:
        for mm in uniq:
            idx = [i for i, x in enumerate(methods) if x == mm]
            ax.scatter(dm[idx], y[idx], s=36, color=cby[mm], label=mm, alpha=0.8, edgecolors="none")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dm_2s (local-correctness violation)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("OffCEM-2s metric")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Objective-1 <-> Objective-2 JOIN", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save = Path(out_dir) / "join_scatter.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nScatter saved to {save}")


def main():
    args = parse_args()
    if args.quick:
        args.n_seeds = 2
        args.n_clusters = "10,50"
        args.eps_list = "0.0,0.7"
        args.n_list = "1000"
        args.reward_std_list = "3"
        args.corrupt_p_list = "0.5"
        print("[quick] tiny smoke grid")
    t0 = time()
    if not args.analyze_only:
        run_grid(args)
        print(f"\nGrid done in {(time()-t0)/60:.1f} min")
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
