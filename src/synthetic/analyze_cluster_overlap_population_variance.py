"""Compute exact population score variance for cluster-overlap results.

The overlap sweep's empirical variance is estimated from a small number of
seeds. This analyzer recomputes each seed/overlap fitted model and evaluates
the OffCEM score variance exactly over the fixed synthetic user-action
population, including continuous reward noise.

Run from ``src/synthetic``:

    conda run -n offcem python -m analyze_cluster_overlap_population_variance \
      --results-dir /Users/cispa/Documents/OffCEM/cluster_overlap_results
"""
import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.ope import RegressionModel
from scipy.stats import rankdata
from sklearn.neural_network import MLPRegressor as MLP
import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from ope import PairWiseRegression
from ope import make_pairwise_data
from run_cluster_overlap_sweep import DEFAULTS
from run_cluster_overlap_sweep import build_cluster_favoring_target
from run_cluster_overlap_sweep import cluster_policy_mass
from run_cluster_overlap_sweep import degrade_behavior_cluster_overlap
from run_cluster_overlap_sweep import load_tidy_csv
from run_cluster_overlap_sweep import make_dataset
from run_cluster_overlap_sweep import parse_overlap_list
from run_cluster_overlap_sweep import policy_value
from run_cluster_overlap_sweep import reconstruct_pi0_population
from run_cluster_overlap_sweep import resample_logged_data
from run_cluster_overlap_sweep import write_csv


def dense_cluster_index_matrix(clusters: np.ndarray) -> np.ndarray:
    ref_mat = np.tile(np.arange(clusters.shape[-1]), (clusters.shape[1], 1))
    cluster_idx_mat = np.zeros_like(clusters[:, :, 0]).astype(int)
    for i, clusters_ in enumerate(clusters):
        cluster_idx_mat[i] = (ref_mat * clusters_).sum(1)
    return rankdata(cluster_idx_mat, method="dense", axis=1) - 1


def train_pairwise_population_predictions(
    bandit_data: Dict,
    cluster_idx_mat: np.ndarray,
    n_clusters: int,
    lr: float = 1e-2,
    batch_size: int = 128,
    num_epochs: int = 30,
    gamma: float = 0.95,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    pairwise_dataset = make_pairwise_data(bandit_data, cluster_idx_mat, n_clusters)
    data_loader = torch.utils.data.DataLoader(
        pairwise_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = PairWiseRegression(
        n_actions=bandit_data["n_actions"],
        n_clusters=n_clusters,
        x_dim=bandit_data["context"].shape[1],
    )
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    model.train()
    for _ in range(num_epochs):
        for x, _c, a1, a2, _e1, _e2, r1, r2 in data_loader:
            loss = model(x, a1, a2, r1, r2)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    x_population = torch.from_numpy(bandit_data["fixed_user_contexts"]).float()
    return model.rel_reward_pred(x_population).detach().numpy()


def train_two_stage_population_predictions(
    bandit_data: Dict,
    clusters: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Train the standard two-stage OffCEM model, predict on all fixed users."""
    cluster_idx_mat = dense_cluster_index_matrix(clusters)
    n_clusters = np.unique(cluster_idx_mat).shape[0]
    h_hat_population = train_pairwise_population_predictions(
        bandit_data,
        cluster_idx_mat,
        n_clusters,
    )
    h_hat_logged = h_hat_population[bandit_data["user_idx"]]
    reward_residual = bandit_data["reward"].astype(float)
    reward_residual -= h_hat_logged[
        np.arange(bandit_data["context"].shape[0]),
        bandit_data["action"],
    ]

    reg_model = RegressionModel(
        n_actions=n_clusters,
        action_context=np.eye(n_clusters),
        base_model=MLP(hidden_layer_sizes=(50, 50, 50), random_state=random_state),
    )
    observed_cluster = cluster_idx_mat[
        bandit_data["user_idx"],
        bandit_data["action"],
    ]
    reg_model.fit(
        context=bandit_data["context"],
        action=observed_cluster,
        reward=reward_residual,
    )
    g_hat_population = reg_model.predict(
        context=bandit_data["fixed_user_contexts"],
    )[:, :, 0]

    f_hat_population = h_hat_population.copy()
    for user in range(f_hat_population.shape[0]):
        f_hat_population[user] += g_hat_population[user][cluster_idx_mat[user]]
    return f_hat_population[:, :, np.newaxis]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact population variance for cluster-overlap OffCEM scores"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/cluster_overlap_results",
    )
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-actions", type=int, default=DEFAULTS["N_ACTIONS"])
    parser.add_argument("--n-users", type=int, default=DEFAULTS["N_USERS"])
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=DEFAULTS["BETA"])
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--target-strength", type=float, default=1.0)
    parser.add_argument(
        "--overlap-list",
        type=str,
        default=None,
        help="Defaults to overlap levels found in cluster_overlap_sweep_tidy.csv",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def score_moments(
    q_population: np.ndarray,
    f_hat: np.ndarray,
    behavior_population: np.ndarray,
    target_population: np.ndarray,
    clusters: np.ndarray,
    reward_std: float,
    n_rounds: int,
    support_tol: float = 1e-12,
) -> Dict[str, float]:
    """Exact moments of the OffCEM one-step score under X,A,R.

    Z = w(X,C(A)) * (R - f(X,A)) + sum_a pi(a|X) f(X,a)

    For unsupported target cluster mass, the score distribution under the
    behavior policy is still finite because those clusters are never sampled,
    but the estimator is outside the usual common-support interpretation.
    """
    q = np.asarray(q_population, dtype=float)
    f = np.asarray(f_hat, dtype=float)
    if f.ndim == 3:
        f = f[:, :, 0]
    behavior = np.asarray(behavior_population, dtype=float)
    target = np.asarray(target_population, dtype=float)

    pi_b_c = cluster_policy_mass(behavior, clusters)
    pi_e_c = cluster_policy_mass(target, clusters)
    supported = pi_b_c > support_tol
    unsupported = (pi_b_c <= support_tol) & (pi_e_c > support_tol)
    cluster_weights = np.divide(
        pi_e_c,
        pi_b_c,
        out=np.zeros_like(pi_e_c),
        where=supported,
    )
    action_weights = cluster_weights[:, clusters]

    residual = q - f
    dm_term = np.sum(target * f, axis=1)
    weighted_residual_mean = np.sum(behavior * action_weights * residual, axis=1)

    mean_by_context = dm_term + weighted_residual_mean
    second_by_context = (
        dm_term**2
        + 2.0 * dm_term * weighted_residual_mean
        + np.sum(
            behavior
            * action_weights**2
            * (residual**2 + reward_std**2),
            axis=1,
        )
    )
    score_mean = float(mean_by_context.mean())
    score_second = float(second_by_context.mean())
    score_variance = max(score_second - score_mean**2, 0.0)
    true_value = policy_value(q, target)
    bias = score_mean - true_value

    return {
        "exact_score_mean": score_mean,
        "exact_true_value": true_value,
        "exact_score_bias": float(bias),
        "exact_score_bias2": float(bias**2),
        "exact_score_variance": float(score_variance),
        "exact_estimator_variance": float(score_variance / n_rounds),
        "exact_estimator_mse": float(bias**2 + score_variance / n_rounds),
        "exact_support_failure": bool(np.any(unsupported)),
        "exact_unsupported_target_cluster_mass": float(
            pi_e_c[unsupported].sum() / pi_e_c.shape[0]
        ),
        "exact_population_cluster_weight_max": float(
            np.max(
                np.divide(
                    pi_e_c,
                    pi_b_c,
                    out=np.full_like(pi_e_c, np.inf),
                    where=supported,
                )
            )
        ),
    }


def infer_seed_indices(rows: List[Dict], n_seeds: int = None) -> List[int]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if n_seeds is not None:
        return list(range(n_seeds))
    return seeds


def infer_overlap_levels(rows: List[Dict], overlap_list: str = None) -> List[float]:
    if overlap_list is not None:
        return parse_overlap_list(overlap_list)
    return sorted({float(row["overlap_level"]) for row in rows}, reverse=True)


def recompute_seed_rows(
    seed_index: int,
    overlap_list: Iterable[float],
    n_rounds: int,
    n_users: int,
    n_actions: int,
    n_clusters: int,
    beta: float,
    reward_std: float,
    target_tau: float,
    target_strength: float,
) -> List[Dict]:
    seed = DEFAULTS["RANDOM_STATE"] + seed_index
    dataset = make_dataset(
        seed=seed,
        n_actions=n_actions,
        beta=beta,
        reward_std=reward_std,
    )
    base_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
        n_users=n_users,
        n_clusters=n_clusters,
        clustering_method="original",
        cluster_balance="natural",
        cluster_temperature=10.0,
    )
    q_population = base_data["fixed_expected_rewards"]
    clusters = base_data["cluster_indices"]
    clusters_3d = clusters_to_onehot_3d(clusters, base_data["n_users"])
    pi0_population = reconstruct_pi0_population(q_population, beta=beta)
    np.testing.assert_allclose(
        pi0_population[base_data["user_idx"]],
        base_data["pi_b"][:, :, 0],
        atol=1e-10,
    )
    target_population = build_cluster_favoring_target(
        q_population=q_population,
        pi0_population=pi0_population,
        clusters=clusters,
        tau=target_tau,
        strength=target_strength,
    )

    rows = []
    for overlap_level in overlap_list:
        behavior_population = degrade_behavior_cluster_overlap(
            pi0_population=pi0_population,
            target_population=target_population,
            clusters=clusters,
            overlap_level=overlap_level,
        )
        bandit_data = resample_logged_data(
            base_data=base_data,
            behavior_population=behavior_population,
            reward_std=reward_std,
            sample_seed=seed + int(round(overlap_level * 100000)) + 9001,
        )
        model_seed = seed + int(round(overlap_level * 100000)) + n_rounds
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        f_x_a = train_two_stage_population_predictions(
            bandit_data=bandit_data,
            clusters=clusters_3d,
            random_state=model_seed,
        )
        row = {
            "seed": int(seed_index),
            "partition": "matched",
            "overlap_level": float(overlap_level),
            "target_tau": float(target_tau),
            "target_strength": float(target_strength),
        }
        row.update(
            score_moments(
                q_population=q_population,
                f_hat=f_x_a,
                behavior_population=behavior_population,
                target_population=target_population,
                clusters=clusters,
                reward_std=reward_std,
                n_rounds=n_rounds,
            )
        )
        rows.append(row)
    return rows


def aggregate_exact_rows(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["overlap_level"], []).append(row)

    aggregates = []
    for overlap_level, items in sorted(grouped.items(), reverse=True):
        out = {
            "partition": "matched",
            "overlap_level": float(overlap_level),
            "target_tau": float(items[0]["target_tau"]),
            "target_strength": float(items[0]["target_strength"]),
            "n_seeds": len(items),
        }
        for key in (
            "exact_true_value",
            "exact_score_mean",
            "exact_score_bias",
            "exact_score_bias2",
            "exact_score_variance",
            "exact_estimator_variance",
            "exact_estimator_mse",
            "exact_unsupported_target_cluster_mass",
            "exact_population_cluster_weight_max",
        ):
            out[f"{key}_mean"] = float(np.mean([item[key] for item in items]))
        out["exact_support_failure_rate"] = float(
            np.mean([item["exact_support_failure"] for item in items])
        )
        aggregates.append(out)
    return aggregates


def merge_with_empirical(
    exact_aggregates: List[Dict],
    empirical_path: Path,
) -> List[Dict]:
    if not empirical_path.exists():
        return exact_aggregates
    with open(empirical_path) as file:
        empirical = {
            float(row["overlap_level"]): row
            for row in csv.DictReader(file)
        }
    merged = []
    for row in exact_aggregates:
        overlap = row["overlap_level"]
        out = dict(row)
        if overlap in empirical:
            empirical_row = empirical[overlap]
            for key in (
                "variance_offcem",
                "mse_offcem",
                "bias2_offcem",
                "cluster_ess_fraction_mean",
                "population_cluster_weight_max_mean",
                "population_cluster_weight_variance_mean",
                "unsupported_target_cluster_mass_mean",
                "complete_support_failure_rate",
            ):
                if key in empirical_row:
                    out[f"empirical_{key}"] = float(empirical_row[key])
        merged.append(out)
    return merged


def plot_variance(rows: List[Dict], out_dir: Path) -> None:
    x = [row["overlap_level"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [row["exact_estimator_variance_mean"] for row in rows],
        marker="o",
        label="exact population Var(Z)/n",
    )
    if "empirical_variance_offcem" in rows[0]:
        ax.plot(
            x,
            [row["empirical_variance_offcem"] for row in rows],
            marker="o",
            label="10-seed empirical variance",
        )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("behavior mass multiplier on target-favored clusters")
    ax.set_ylabel("OffCEM variance")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "exact_population_variance_vs_overlap.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    tidy_path = results_dir / "cluster_overlap_sweep_tidy.csv"
    if not tidy_path.exists():
        raise FileNotFoundError(f"missing tidy results: {tidy_path}")

    tidy_rows = load_tidy_csv(tidy_path)
    seed_indices = infer_seed_indices(tidy_rows, args.n_seeds)
    overlap_list = infer_overlap_levels(tidy_rows, args.overlap_list)

    exact_rows = []
    for seed_index in tqdm(seed_indices, desc="exact-variance seeds"):
        exact_rows.extend(
            recompute_seed_rows(
                seed_index=seed_index,
                overlap_list=overlap_list,
                n_rounds=args.n_rounds,
                n_users=args.n_users,
                n_actions=args.n_actions,
                n_clusters=args.n_clusters,
                beta=args.beta,
                reward_std=args.reward_std,
                target_tau=args.target_tau,
                target_strength=args.target_strength,
            )
        )
    exact_path = results_dir / "cluster_overlap_exact_variance_tidy.csv"
    aggregate_path = results_dir / "cluster_overlap_exact_variance_aggregate.csv"
    merged_path = results_dir / "cluster_overlap_exact_vs_empirical_variance.csv"
    aggregate_rows = aggregate_exact_rows(exact_rows)
    merged_rows = merge_with_empirical(
        aggregate_rows,
        results_dir / "cluster_overlap_sweep_aggregate.csv",
    )
    write_csv(exact_path, exact_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(merged_path, merged_rows)
    if not args.no_plot:
        plot_variance(merged_rows, results_dir)
    print(f"wrote {exact_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {merged_path}")
    if not args.no_plot:
        print(f"wrote {results_dir / 'exact_population_variance_vs_overlap.png'}")


if __name__ == "__main__":
    main()
