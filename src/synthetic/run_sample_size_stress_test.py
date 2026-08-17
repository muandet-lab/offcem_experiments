"""Focused sample-size stress test for OffCEM.

The experiment fixes one synthetic world per seed, generates a single logged
stream of length max(n_list), and evaluates nested prefixes. Only logged sample
size varies within a seed.

Run from ``src/synthetic``:

    conda run -n offcem python -m run_sample_size_stress_test --quick
"""
import argparse
import json
import warnings
from pathlib import Path
from time import time
from typing import Dict
from typing import Iterable
from typing import List

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obp.dataset import linear_reward_function
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import RegressionModel
from sklearn.neural_network import MLPRegressor as MLP
import torch
from tqdm import tqdm

from clustering import clusters_to_onehot_3d
from clustering import compute_clusters
from dataset import SyntheticBanditDataset
from ope import OffCEM
from ope import OffPolicyEvaluation
from ope import train_reward_model_via_two_stage
from policy import gen_eps_greedy


DEFAULTS = dict(
    DIM_CONTEXT=10,
    N_USERS=200,
    N_ACTIONS=1000,
    N_CAT_PER_DIM=5,
    LATENT_PARAM_MAT_DIM=5,
    N_CAT_DIM=10,
    BETA=-0.1,
    N_DEF_ACTIONS=0.0,
    REWARD_TYPE="continuous",
    RANDOM_STATE=12345,
    ESTIMATION_SEED_OFFSET=100000,
)

ROUND_LEVEL_FIELDS = (
    "user_idx",
    "context",
    "action",
    "reward",
    "expected_reward",
    "pi_b",
    "pscore",
    "action_embed",
)

WORLD_LEVEL_FIELDS = (
    "n_users",
    "n_actions",
    "clusters",
    "cluster_indices",
    "action_context",
    "action_context_one_hot",
    "fixed_user_contexts",
    "fixed_expected_rewards",
    "g_x_e",
    "p_e_a",
    "position",
)

ESTIMATOR_LABELS = {
    "matched": "OffCEM matched",
    "reestimated_wfss": "OffCEM reestimated WFSS",
    "kmeans": "OffCEM K-means",
    "DR": "DR",
    "DM": "DM",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample-size stress test with one fixed synthetic world per seed"
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="500,1000,3000,10000,30000,100000",
        help="Comma-separated logged prefix sizes",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--reward-std", type=float, default=3.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/cispa/Documents/OffCEM/sample_size_stress_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use n=500,3000 and 2 seeds",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate CSV and plots from tidy results",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_n_list(value: str) -> List[int]:
    n_list = [int(item.replace("_", "")) for item in value.split(",") if item]
    if not n_list:
        raise ValueError("--n-list must contain at least one value")
    if any(n <= 0 for n in n_list):
        raise ValueError("all n values must be positive")
    return sorted(set(n_list))


def make_dataset(seed: int, reward_std: float) -> SyntheticBanditDataset:
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
        random_state=seed,
    )


def population_policy_value(
    fixed_expected_rewards: np.ndarray,
    eps: float,
) -> tuple[float, np.ndarray]:
    pi_e_population = gen_eps_greedy(
        expected_reward=fixed_expected_rewards,
        eps=eps,
    )
    true_value = float(
        np.average(
            fixed_expected_rewards,
            weights=pi_e_population[:, :, 0],
            axis=1,
        ).mean()
    )
    return true_value, pi_e_population


def build_feedback_prefix(full_bandit_data: Dict, n: int) -> Dict:
    """Build a nested logged-data prefix while preserving the synthetic world.

    Round-level fields are sliced to the first ``n`` interactions. World-level
    fields are reused. User-action reward matrices are rebuilt from the prefix
    with mean aggregation for repeated observations.
    """
    if n <= 0 or n > int(full_bandit_data["n_rounds"]):
        raise ValueError("n must be in [1, full_bandit_data['n_rounds']]")

    prefix = {"n_rounds": int(n)}
    for key in WORLD_LEVEL_FIELDS:
        if key in full_bandit_data:
            prefix[key] = full_bandit_data[key]
    for key in ROUND_LEVEL_FIELDS:
        if key not in full_bandit_data:
            continue
        value = full_bandit_data[key]
        if value is None:
            prefix[key] = None
        else:
            prefix[key] = value[:n].copy()

    n_users = int(prefix["n_users"])
    n_actions = int(prefix["n_actions"])
    reward_sum_mat = np.zeros((n_users, n_actions), dtype=float)
    obs_count_mat = np.zeros((n_users, n_actions), dtype=int)
    for user, action, reward in zip(
        prefix["user_idx"],
        prefix["action"],
        prefix["reward"],
    ):
        reward_sum_mat[int(user), int(action)] += float(reward)
        obs_count_mat[int(user), int(action)] += 1

    reward_mat = np.zeros((n_users, n_actions), dtype=float)
    observed = obs_count_mat > 0
    reward_mat[observed] = reward_sum_mat[observed] / obs_count_mat[observed]

    prefix["reward_sum_mat"] = reward_sum_mat
    prefix["obs_count_mat"] = obs_count_mat
    prefix["reward_mat"] = reward_mat
    prefix["obs_mat"] = observed.astype(int)

    _assert_prefix_shapes(prefix, n)
    if int(prefix["obs_count_mat"].sum()) != n:
        raise AssertionError("obs_count_mat.sum() must equal prefix length")
    return prefix


def _assert_prefix_shapes(prefix: Dict, n: int) -> None:
    for key in ROUND_LEVEL_FIELDS:
        if key not in prefix or prefix[key] is None:
            continue
        if prefix[key].shape[0] != n:
            raise AssertionError(
                f"{key}.shape[0] must be {n}, got {prefix[key].shape[0]}"
            )


def build_fixed_estimation_partitions(
    full_bandit_data: Dict,
    n_clusters: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    estimation_seed = seed + DEFAULTS["ESTIMATION_SEED_OFFSET"]
    reestimated = compute_clusters(
        action_features=full_bandit_data["action_context_one_hot"],
        n_clusters=n_clusters,
        method="original",
        balance="natural",
        random_state=estimation_seed,
        temperature=10.0,
    )
    kmeans = compute_clusters(
        action_features=full_bandit_data["action_context_one_hot"],
        n_clusters=n_clusters,
        method="kmeans",
        balance="natural",
        random_state=estimation_seed,
    )
    return {
        "matched": full_bandit_data["cluster_indices"].copy(),
        "reestimated_wfss": reestimated,
        "kmeans": kmeans,
    }


def fit_action_reward_model(
    bandit_data: Dict,
    random_state: int,
) -> np.ndarray:
    reg_model = RegressionModel(
        n_actions=bandit_data["n_actions"],
        action_context=bandit_data["action_context_one_hot"],
        base_model=MLP(
            hidden_layer_sizes=(50, 50, 50),
            random_state=random_state,
        ),
    )
    return reg_model.fit_predict(
        context=bandit_data["context"],
        action=bandit_data["action"],
        reward=bandit_data["reward"],
    )


def estimate_dr_dm(
    bandit_data: Dict,
    pi_e_logged: np.ndarray,
    q_x_a: np.ndarray,
) -> Dict[str, float]:
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=[
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
        ],
    )
    return ope.estimate_policy_values(
        action_dist=pi_e_logged,
        estimated_rewards_by_reg_model={
            "DR": q_x_a,
            "DM": q_x_a,
        },
    )


def estimate_offcem(
    bandit_data: Dict,
    pi_e_logged: np.ndarray,
    clusters_3d: np.ndarray,
    f_x_a: np.ndarray,
    estimator_name: str,
) -> float:
    observed_clusters = clusters_3d[bandit_data["user_idx"]]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=[
            OffCEM(
                n_actions=bandit_data["n_actions"],
                estimator_name=estimator_name,
                is_clustering=True,
            )
        ],
    )
    values = ope.estimate_policy_values(
        action_dist=pi_e_logged,
        action_embed=bandit_data["action_embed"],
        pi_b=bandit_data["pi_b"],
        p_e_a={estimator_name: observed_clusters},
        estimated_rewards_by_reg_model={estimator_name: f_x_a},
    )
    return float(values[estimator_name])


def run_seed(
    seed_index: int,
    n_list: List[int],
    n_clusters: int,
    eps: float,
    reward_std: float,
) -> List[Dict]:
    seed = DEFAULTS["RANDOM_STATE"] + seed_index
    dataset = make_dataset(seed=seed, reward_std=reward_std)
    full_data = dataset.obtain_batch_bandit_feedback(
        n_rounds=max(n_list),
        n_users=DEFAULTS["N_USERS"],
        n_clusters=n_clusters,
        clustering_method="original",
        cluster_balance="natural",
        cluster_temperature=10.0,
    )
    true_value, pi_e_population = population_policy_value(
        full_data["fixed_expected_rewards"],
        eps=eps,
    )
    partitions = build_fixed_estimation_partitions(
        full_bandit_data=full_data,
        n_clusters=n_clusters,
        seed=seed,
    )
    _assert_fixed_world(full_data, true_value, partitions)

    rows = []
    previous_prefixes = {}
    cluster_tensors = {
        name: clusters_to_onehot_3d(labels, full_data["n_users"])
        for name, labels in partitions.items()
    }
    for n in n_list:
        prefix = build_feedback_prefix(full_data, n)
        _assert_world_unchanged(prefix, full_data, true_value, eps)
        _assert_nested_prefix(prefix, previous_prefixes)
        previous_prefixes[n] = prefix

        pi_e_logged = pi_e_population[prefix["user_idx"]]
        model_seed = seed + n
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        q_x_a = fit_action_reward_model(prefix, random_state=model_seed)
        base_estimates = estimate_dr_dm(prefix, pi_e_logged, q_x_a)
        for estimator in ("DR", "DM"):
            rows.append(
                _result_row(
                    n=n,
                    seed_index=seed_index,
                    estimator=estimator,
                    estimate=float(base_estimates[estimator]),
                    true_value=true_value,
                )
            )

        for method in ("matched", "reestimated_wfss", "kmeans"):
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            f_x_a = train_reward_model_via_two_stage(
                bandit_data=prefix,
                clusters=cluster_tensors[method],
                need_q_x_a=False,
                random_state=model_seed,
            )
            estimator_name = ESTIMATOR_LABELS[method]
            estimate = estimate_offcem(
                bandit_data=prefix,
                pi_e_logged=pi_e_logged,
                clusters_3d=cluster_tensors[method],
                f_x_a=f_x_a,
                estimator_name=estimator_name,
            )
            rows.append(
                _result_row(
                    n=n,
                    seed_index=seed_index,
                    estimator=estimator_name,
                    estimate=estimate,
                    true_value=true_value,
                )
            )
    return rows


def _result_row(n, seed_index, estimator, estimate, true_value):
    error = float(estimate - true_value)
    return {
        "n": int(n),
        "seed": int(seed_index),
        "estimator": estimator,
        "estimate": float(estimate),
        "true_value": float(true_value),
        "squared_error": float(error**2),
    }


def _assert_fixed_world(
    full_data: Dict,
    true_value: float,
    partitions: Dict[str, np.ndarray],
) -> None:
    if not np.isfinite(true_value):
        raise AssertionError("true value must be finite")
    expected = full_data["fixed_expected_rewards"]
    action_features = full_data["action_context_one_hot"]
    clusters = full_data["cluster_indices"]
    if expected.ndim != 2 or action_features.ndim != 2 or clusters.ndim != 1:
        raise AssertionError("fixed world arrays have invalid dimensions")
    for labels in partitions.values():
        if labels.shape != clusters.shape:
            raise AssertionError("partition labels must have one entry per action")


def _assert_world_unchanged(
    prefix: Dict,
    full_data: Dict,
    true_value: float,
    eps: float,
) -> None:
    np.testing.assert_array_equal(
        prefix["fixed_expected_rewards"],
        full_data["fixed_expected_rewards"],
    )
    np.testing.assert_array_equal(
        prefix["action_context_one_hot"],
        full_data["action_context_one_hot"],
    )
    np.testing.assert_array_equal(
        prefix["cluster_indices"],
        full_data["cluster_indices"],
    )
    check_value, _ = population_policy_value(prefix["fixed_expected_rewards"], eps)
    if not np.isclose(check_value, true_value):
        raise AssertionError("V_true changed across prefixes")


def _assert_nested_prefix(prefix: Dict, previous_prefixes: Dict[int, Dict]) -> None:
    for smaller_n, smaller in previous_prefixes.items():
        if smaller_n > prefix["n_rounds"]:
            continue
        for key in ROUND_LEVEL_FIELDS:
            if key not in prefix or prefix[key] is None:
                continue
            np.testing.assert_array_equal(prefix[key][:smaller_n], smaller[key])


def aggregate_results(rows: Iterable[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["n"], row["estimator"]), []).append(row)

    aggregates = []
    for (n, estimator), items in sorted(grouped.items()):
        estimates = np.array([item["estimate"] for item in items], dtype=float)
        true_values = np.array([item["true_value"] for item in items], dtype=float)
        errors = estimates - true_values
        mse = float(np.mean(errors**2))
        bias2 = float(np.mean(errors) ** 2)
        variance = float(np.mean((errors - errors.mean()) ** 2))
        norm = float(np.mean(true_values**2))
        aggregates.append(
            {
                "n": int(n),
                "estimator": estimator,
                "mse": mse,
                "rel_mse": mse / max(norm, 1e-12),
                "bias2": bias2,
                "variance": variance,
                "estimate_mean": float(estimates.mean()),
                "true_value_mean": float(true_values.mean()),
                "n_seeds": len(items),
            }
        )
    return aggregates


def write_csv(path: Path, rows: List[Dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_tidy_csv(path: Path) -> List[Dict]:
    import csv

    with open(path) as file:
        return [
            {
                "n": int(row["n"]),
                "seed": int(row["seed"]),
                "estimator": row["estimator"],
                "estimate": float(row["estimate"]),
                "true_value": float(row["true_value"]),
                "squared_error": float(row["squared_error"]),
            }
            for row in csv.DictReader(file)
        ]


def plot_aggregates(aggregates: List[Dict], out_dir: Path) -> None:
    metrics = [
        ("rel_mse", "Relative MSE", "rel_mse_vs_n.png"),
        ("bias2", "Bias^2", "bias2_vs_n.png"),
        ("variance", "Variance", "variance_vs_n.png"),
    ]
    estimators = [
        "OffCEM matched",
        "OffCEM reestimated WFSS",
        "OffCEM K-means",
        "DR",
        "DM",
    ]
    colors = {
        "OffCEM matched": "#1f77b4",
        "OffCEM reestimated WFSS": "#ff7f0e",
        "OffCEM K-means": "#2ca02c",
        "DR": "#d62728",
        "DM": "#9467bd",
    }
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for estimator in estimators:
            selected = [
                row for row in aggregates if row["estimator"] == estimator
            ]
            if not selected:
                continue
            selected = sorted(selected, key=lambda row: row["n"])
            ax.plot(
                [row["n"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                linewidth=1.4,
                markersize=4,
                label=estimator,
                color=colors.get(estimator),
            )
        ax.set_xscale("log")
        ax.set_xlabel("logged interactions n")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_experiment(args) -> None:
    if args.quick:
        args.n_list = "500,3000"
        args.n_seeds = 2
        print("[quick] n-list=500,3000 n-seeds=2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = out_dir / "sample_size_stress_tidy.csv"
    aggregate_path = out_dir / "sample_size_stress_aggregate.csv"

    if args.plot_only:
        rows = load_tidy_csv(tidy_path)
        aggregates = aggregate_results(rows)
        write_csv(aggregate_path, aggregates)
        if not args.no_plot:
            plot_aggregates(aggregates, out_dir)
        return

    n_list = parse_n_list(args.n_list)
    all_rows = []
    started = time()
    for seed_index in tqdm(range(args.n_seeds), desc="sample-size seeds"):
        all_rows.extend(
            run_seed(
                seed_index=seed_index,
                n_list=n_list,
                n_clusters=args.n_clusters,
                eps=args.eps,
                reward_std=args.reward_std,
            )
        )
        with open(out_dir / "latest_rows.json", "w") as file:
            json.dump(all_rows, file, indent=2)

    aggregates = aggregate_results(all_rows)
    write_csv(tidy_path, all_rows)
    write_csv(aggregate_path, aggregates)
    if not args.no_plot:
        plot_aggregates(aggregates, out_dir)
    elapsed = (time() - started) / 60
    print(f"wrote {tidy_path}")
    print(f"wrote {aggregate_path}")
    if not args.no_plot:
        print(f"wrote plots to {out_dir}")
    print(f"done in {elapsed:.1f} min")


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
