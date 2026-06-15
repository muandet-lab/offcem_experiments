"""Estimator orchestration for the OffCEM applicability benchmark."""
from typing import Dict

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPS
from obp.ope import RegressionModel
from sklearn.neural_network import MLPRegressor as MLP

from ope import OffCEM
from ope import OffPolicyEvaluation
from ope import train_reward_model_via_two_stage


BASELINE_NAMES = ("IPS", "SNIPS", "DR", "DM", "MIPS-embedding")


def fit_action_reward_model(bandit_data: Dict, random_state: int) -> np.ndarray:
    model = RegressionModel(
        n_actions=bandit_data["n_actions"],
        action_context=bandit_data["action_context_one_hot"],
        base_model=MLP(
            hidden_layer_sizes=(50, 50, 50),
            random_state=random_state,
        ),
    )
    return model.fit_predict(
        context=bandit_data["context"],
        action=bandit_data["action"],
        reward=bandit_data["reward"],
    )


def estimate_shared_baselines(
    bandit_data: Dict,
    pi_e: np.ndarray,
    q_x_a: np.ndarray,
) -> Dict[str, float]:
    n_actions = bandit_data["n_actions"]
    estimators = [
        IPS(estimator_name="IPS"),
        SNIPS(estimator_name="SNIPS"),
        DR(estimator_name="DR"),
        DM(estimator_name="DM"),
        OffCEM(n_actions=n_actions, estimator_name="MIPS-embedding"),
    ]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=estimators,
    )
    return ope.estimate_policy_values(
        action_dist=pi_e,
        action_embed=bandit_data["action_embed"],
        pi_b=bandit_data["pi_b"],
        p_e_a={"MIPS-embedding": bandit_data["p_e_a"]},
        estimated_rewards_by_reg_model={
            "DR": q_x_a,
            "DM": q_x_a,
            "MIPS-embedding": np.zeros_like(q_x_a),
        },
    )


def estimate_partition_methods(
    bandit_data: Dict,
    pi_e: np.ndarray,
    clusters_3d: np.ndarray,
    f_x_a: np.ndarray,
    q_x_a: np.ndarray,
    label: str,
) -> Dict[str, float]:
    n_actions = bandit_data["n_actions"]
    names = {
        "offcem_2s": f"OffCEM-2s::{label}",
        "offcem_1s": f"OffCEM-1s::{label}",
        "cluster_ips": f"Cluster-IPS::{label}",
    }
    estimators = [
        OffCEM(
            n_actions=n_actions,
            estimator_name=names["offcem_2s"],
            is_clustering=True,
        ),
        OffCEM(
            n_actions=n_actions,
            estimator_name=names["offcem_1s"],
            is_clustering=True,
        ),
        OffCEM(
            n_actions=n_actions,
            estimator_name=names["cluster_ips"],
            is_clustering=True,
        ),
    ]
    observed_clusters = clusters_3d[bandit_data["user_idx"]]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=estimators,
    )
    return ope.estimate_policy_values(
        action_dist=pi_e,
        action_embed=bandit_data["action_embed"],
        pi_b=bandit_data["pi_b"],
        p_e_a={name: observed_clusters for name in names.values()},
        estimated_rewards_by_reg_model={
            names["offcem_2s"]: f_x_a,
            names["offcem_1s"]: q_x_a,
            names["cluster_ips"]: np.zeros_like(q_x_a),
        },
    )


def fit_two_stage_model(
    bandit_data: Dict,
    clusters_3d: np.ndarray,
    random_state: int,
) -> np.ndarray:
    return train_reward_model_via_two_stage(
        bandit_data=bandit_data,
        clusters=clusters_3d,
        need_q_x_a=False,
        random_state=random_state,
    )

