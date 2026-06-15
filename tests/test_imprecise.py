import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from imprecise.clustering import fit_imprecise_clustering  # noqa: E402
from imprecise.clustering import sample_partitions  # noqa: E402
from imprecise.controlled_overlap import align_memberships  # noqa: E402
from imprecise.controlled_overlap import ControlledOverlapDataset  # noqa: E402
from imprecise.controlled_overlap import membership_diagnostics  # noqa: E402
from imprecise.domain_outcome import build_partitions  # noqa: E402
from imprecise.domain_outcome import DomainOutcomeDataset  # noqa: E402
from imprecise.relaxed_local_correctness import (  # noqa: E402
    relaxed_local_correctness_bound,
)


class ImpreciseClusteringTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.features = np.vstack(
            [
                rng.normal(-2.0, 0.2, size=(12, 3)),
                rng.normal(0.0, 0.2, size=(12, 3)),
                rng.normal(2.0, 0.2, size=(12, 3)),
            ]
        )

    def test_fcm_memberships_are_normalized(self):
        result = fit_imprecise_clustering(
            self.features, 3, "fcm", random_state=10
        )
        self.assertEqual(result.scores.shape, (36, 3))
        np.testing.assert_allclose(result.scores.sum(axis=1), 1.0)
        self.assertEqual(np.unique(result.labels).size, 3)

    def test_pcm_typicalities_are_bounded(self):
        result = fit_imprecise_clustering(
            self.features, 3, "pcm", random_state=10
        )
        self.assertTrue(np.all(result.scores >= 0.0))
        self.assertTrue(np.all(result.scores <= 1.0))
        self.assertIn("max_typicality_mean", result.uncertainty)
        self.assertIn("coincident_center_pairs", result.uncertainty)
        self.assertEqual(np.unique(result.labels).size, 3)

    def test_rough_candidates_and_samples_are_valid(self):
        result = fit_imprecise_clustering(
            self.features,
            3,
            "rough",
            random_state=10,
            rough_ratio=1.4,
        )
        self.assertEqual(len(result.candidates), len(self.features))
        partitions = sample_partitions(result, 5, random_state=20)
        self.assertEqual(len(partitions), 5)
        for labels in partitions:
            self.assertEqual(labels.shape, (36,))
            self.assertEqual(np.unique(labels).size, 3)

    def test_fcm_partition_sampling_preserves_clusters(self):
        result = fit_imprecise_clustering(
            self.features, 3, "fcm", random_state=10
        )
        partitions = sample_partitions(result, 5, random_state=20)
        for labels in partitions:
            self.assertEqual(np.unique(labels).size, 3)

    def test_ecm_dependency_error_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "evclust"):
            fit_imprecise_clustering(
                self.features, 3, "ecm", random_state=10
            )

    def test_membership_alignment_recovers_permuted_columns(self):
        truth = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.3, 0.7],
            ]
        )
        estimated = truth[:, [2, 0, 1]]
        np.testing.assert_allclose(align_memberships(estimated, truth), truth)

    def test_controlled_overlap_reward_relevance_control(self):
        common = dict(
            n_actions=60,
            n_users=20,
            n_clusters=4,
            dim_context=5,
            dim_action=4,
            ambiguity_fraction=0.5,
            ambiguity_strength=0.45,
            feature_noise=0.05,
            cluster_reward_share=0.8,
            reward_scale=2.0,
            reward_std=1.0,
            beta=-0.1,
            target_epsilon=0.2,
            random_state=123,
        )
        relevant = ControlledOverlapDataset(
            reward_mode="reward_relevant", **common
        ).generate(n_rounds=100, sample_random_state=456)
        control = ControlledOverlapDataset(
            reward_mode="feature_only", **common
        ).generate(n_rounds=100, sample_random_state=456)
        np.testing.assert_allclose(
            relevant["true_memberships"],
            control["true_memberships"],
        )
        self.assertGreater(relevant["reward_mixture_effect"].mean(), 0.0)
        self.assertEqual(control["reward_mixture_effect"].mean(), 0.0)

    def test_membership_diagnostics_detect_known_memberships(self):
        truth = np.array(
            [
                [1.0, 0.0],
                [0.75, 0.25],
                [0.0, 1.0],
                [0.4, 0.6],
            ]
        )
        diagnostics = membership_diagnostics(
            estimated=truth[:, ::-1],
            truth=truth,
            reward_mixture_effect=np.array([0.0, 0.2, 0.0, 0.3]),
        )
        self.assertAlmostEqual(diagnostics["membership_mse"], 0.0)
        self.assertAlmostEqual(diagnostics["primary_accuracy"], 1.0)
        self.assertAlmostEqual(diagnostics["ambiguity_auc"], 1.0)

    def test_domain_outcome_auxiliary_sample_is_independent(self):
        dataset = self._domain_dataset(domain_alignment=0.5)
        evaluation = dataset.generate(100, sample_random_state=200)
        auxiliary = dataset.generate(100, sample_random_state=300)
        np.testing.assert_allclose(
            evaluation["fixed_expected_rewards"],
            auxiliary["fixed_expected_rewards"],
        )
        np.testing.assert_allclose(
            evaluation["domain_features"],
            auxiliary["domain_features"],
        )
        self.assertFalse(np.array_equal(
            evaluation["action"], auxiliary["action"]
        ))

    def test_domain_alignment_changes_reward_not_metadata(self):
        domain = self._domain_dataset(domain_alignment=1.0).generate(
            100, sample_random_state=200
        )
        outcome = self._domain_dataset(domain_alignment=0.0).generate(
            100, sample_random_state=200
        )
        np.testing.assert_allclose(
            domain["domain_features"], outcome["domain_features"]
        )
        np.testing.assert_array_equal(
            domain["domain_labels"], outcome["domain_labels"]
        )
        self.assertFalse(np.allclose(
            domain["fixed_expected_rewards"],
            outcome["fixed_expected_rewards"],
        ))

    def test_domain_outcome_partitions_have_expected_shape(self):
        data = self._domain_dataset(domain_alignment=0.5).generate(
            100, sample_random_state=200
        )
        auxiliary_q = data["expected_reward"][:, :, np.newaxis]
        partitions = build_partitions(
            evaluation_data=data,
            auxiliary_q=auxiliary_q,
            n_clusters=4,
            random_state=400,
            profile_components=3,
        )
        expected = {
            "feature_only",
            "domain_informed",
            "oracle_outcome",
            "estimated_outcome",
            "hybrid",
            "shuffled_domain",
            "random",
        }
        self.assertEqual(set(partitions), expected)
        for labels in partitions.values():
            self.assertEqual(labels.shape, (60,))
            self.assertEqual(np.unique(labels).size, 4)

    @staticmethod
    def _domain_dataset(domain_alignment):
        return DomainOutcomeDataset(
            n_actions=60,
            n_users=20,
            n_clusters=4,
            dim_context=5,
            dim_generic=4,
            dim_domain=4,
            domain_alignment=domain_alignment,
            domain_label_noise=0.25,
            action_noise=0.1,
            reward_scale=2.0,
            reward_std=1.0,
            beta=-0.1,
            target_epsilon=0.2,
            random_state=100,
        )

    def test_relaxed_local_correctness_exact_case_has_zero_bias(self):
        q = np.array([[1.0, 2.0, -1.0, 3.0]])
        prediction = q - np.array([[0.4, 0.4, -0.2, -0.2]])
        pi_b = np.array([[0.4, 0.1, 0.3, 0.2]])
        pi_e = np.array([[0.1, 0.4, 0.2, 0.3]])
        result = relaxed_local_correctness_bound(
            q, prediction, pi_b, pi_e, np.array([0, 0, 1, 1])
        )
        self.assertAlmostEqual(result["population_bias"], 0.0)
        self.assertAlmostEqual(result["epsilon_tv_bound"], 0.0)
        self.assertTrue(result["bound_covers_population_bias"])

    def test_relaxed_local_correctness_bound_covers_bias(self):
        q = np.array(
            [
                [1.0, 2.0, -1.0, 3.0],
                [0.5, 1.5, 2.0, -2.0],
            ]
        )
        prediction = q - np.array(
            [
                [0.0, 1.0, -0.5, 0.5],
                [0.2, -0.4, 0.8, -0.1],
            ]
        )
        pi_b = np.array(
            [
                [0.4, 0.1, 0.3, 0.2],
                [0.1, 0.4, 0.2, 0.3],
            ]
        )
        pi_e = np.array(
            [
                [0.1, 0.4, 0.2, 0.3],
                [0.3, 0.2, 0.4, 0.1],
            ]
        )
        result = relaxed_local_correctness_bound(
            q, prediction, pi_b, pi_e, np.array([0, 0, 1, 1])
        )
        self.assertGreater(result["epsilon_tv_bound"], 0.0)
        self.assertLessEqual(
            abs(result["population_bias"]),
            result["epsilon_tv_bound"] + 1e-10,
        )
        self.assertTrue(result["bound_covers_population_bias"])


if __name__ == "__main__":
    unittest.main()
