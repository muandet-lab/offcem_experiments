"""Fit and evaluate interpretable OffCEM applicability models."""
import json
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


DEFAULT_NUMERIC_FEATURES = [
    "reward.dm_ratio",
    "reward.dm_variance_scaled",
    "reward.dm_policy_weighted",
    "reward.reward_mse_2s",
    "overlap.action_ess_fraction",
    "overlap.cluster_ess_fraction",
    "overlap.action_weight_max",
    "overlap.action_weight_variance",
    "overlap.cluster_weight_max",
    "overlap.cluster_weight_variance",
    "overlap.unsupported_target_mass",
    "overlap.clusters_without_support_fraction",
    "overlap.observed_actions_per_cluster_mean",
    "overlap.pairwise_comparisons_per_round",
    "overlap.cluster_size_cv",
    "partition.ari_generating_estimation",
    "realized_share.cluster",
    "realized_share.feature",
    "realized_share.identity",
    "realized_share.hidden",
    "config.reward.feature_nonlinearity",
    "config.reward.noise_std",
    "config.policies.target_epsilon",
    "config.policies.deficient_action_fraction",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "partition",
    "config.dataset.cluster_balance",
    "config.dataset.gen_clustering_method",
]


def fit_applicability_models(
    train_csvs: List[str],
    test_csvs: List[str],
    output_dir: str,
    baseline: str = "DR",
    minimum_relative_effect: float = 0.10,
) -> Dict:
    train = _load_comparisons(train_csvs, baseline)
    test = _load_comparisons(test_csvs, baseline)
    train = _engineer_features(train)
    test = _engineer_features(test)

    train["target"] = (
        train["relative_effect"] <= -minimum_relative_effect
    ).astype(int)
    test["target"] = (
        test["relative_effect"] <= -minimum_relative_effect
    ).astype(int)
    if train["target"].nunique() < 2:
        raise ValueError("Training data must contain both OffCEM wins and non-wins")

    numeric = [name for name in DEFAULT_NUMERIC_FEATURES if name in train]
    categorical = [
        name for name in DEFAULT_CATEGORICAL_FEATURES if name in train
    ]
    feature_names = numeric + categorical
    train = train.dropna(subset=["target"])
    test = test.dropna(subset=["target"])

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    logistic = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=12345,
                ),
            ),
        ]
    )
    logistic.fit(train[feature_names], train["target"])
    logistic_probability = logistic.predict_proba(test[feature_names])[:, 1]

    tree_features = _tree_frame(train, numeric, categorical)
    tree_test = _tree_frame(test, numeric, categorical, columns=tree_features.columns)
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=max(5, len(train) // 50),
        class_weight="balanced",
        random_state=12345,
    )
    tree.fit(tree_features, train["target"])
    tree_probability = tree.predict_proba(tree_test)[:, 1]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = test[
        ["cell_id", "partition", "baseline", "relative_effect", "target"]
    ].copy()
    predictions["logistic_probability"] = logistic_probability
    predictions["tree_probability"] = tree_probability
    predictions.to_csv(output / "heldout_predictions.csv", index=False)

    metrics = {
        "baseline": baseline,
        "minimum_relative_effect": minimum_relative_effect,
        "n_train": len(train),
        "n_test": len(test),
        "features": feature_names,
        "logistic": _prediction_metrics(test["target"], logistic_probability),
        "tree": _prediction_metrics(test["target"], tree_probability),
    }
    with open(output / "model_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)
    with open(output / "tree_rules.txt", "w") as file:
        file.write(export_text(tree, feature_names=list(tree_features.columns)))
    return metrics


def _load_comparisons(paths: List[str], baseline: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    return data.loc[data["baseline"] == baseline].copy()


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    rounds = data["config.n_rounds"].clip(lower=1)
    data["overlap.action_ess_fraction"] = data["overlap.action_ess"] / rounds
    data["overlap.cluster_ess_fraction"] = data["overlap.cluster_ess"] / rounds
    data["overlap.pairwise_comparisons_per_round"] = (
        data["overlap.pairwise_comparisons"] / rounds
    )
    return data


def _tree_frame(
    data: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
    columns=None,
) -> pd.DataFrame:
    frame = data[numeric].copy()
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    categorical_frame = pd.get_dummies(
        data[categorical].fillna("missing"),
        prefix=categorical,
        dtype=float,
    )
    frame = pd.concat([frame, categorical_frame], axis=1)
    if columns is not None:
        frame = frame.reindex(columns=columns, fill_value=0.0)
    return frame


def _prediction_metrics(target: pd.Series, probability: np.ndarray) -> Dict:
    prediction = (probability >= 0.5).astype(int)
    fraction_positive, mean_prediction = calibration_curve(
        target,
        probability,
        n_bins=min(10, max(2, len(target) // 10)),
        strategy="quantile",
    )
    result = {
        "accuracy": float(accuracy_score(target, prediction)),
        "brier_score": float(brier_score_loss(target, probability)),
        "calibration": [
            {
                "mean_prediction": float(predicted),
                "fraction_positive": float(observed),
            }
            for predicted, observed in zip(mean_prediction, fraction_positive)
        ],
    }
    if target.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(target, probability))
    else:
        result["roc_auc"] = None
    return result

