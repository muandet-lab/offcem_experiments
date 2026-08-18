import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from run_cluster_overlap_sweep import aggregate_results  # noqa: E402
from run_cluster_overlap_sweep import build_cluster_favoring_target  # noqa: E402
from run_cluster_overlap_sweep import cluster_overlap_diagnostics  # noqa: E402
from run_cluster_overlap_sweep import cluster_policy_mass  # noqa: E402
from run_cluster_overlap_sweep import degrade_behavior_cluster_overlap  # noqa: E402
from run_cluster_overlap_sweep import parse_overlap_list  # noqa: E402
from run_cluster_overlap_sweep import policy_value  # noqa: E402
from run_cluster_overlap_sweep import resample_logged_data  # noqa: E402


class ClusterOverlapSweepTest(unittest.TestCase):
    def setUp(self):
        self.q = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 4.0, 5.0],
            ]
        )
        self.pi0 = np.array(
            [
                [0.20, 0.30, 0.10, 0.40],
                [0.15, 0.35, 0.25, 0.25],
            ]
        )
        self.clusters = np.array([0, 0, 1, 1])

    def test_parse_overlap_list_deduplicates_and_sorts_descending(self):
        self.assertEqual(parse_overlap_list("0,.5,1,.5"), [1.0, 0.5, 0.0])
        with self.assertRaises(ValueError):
            parse_overlap_list("1.2")

    def test_cluster_favoring_target_preserves_pi0_conditionals(self):
        target = build_cluster_favoring_target(
            self.q,
            self.pi0,
            self.clusters,
            tau=1.0,
            strength=1.0,
        )
        np.testing.assert_allclose(target.sum(axis=1), 1.0)
        pi0_c = cluster_policy_mass(self.pi0, self.clusters)
        target_c = cluster_policy_mass(target, self.clusters)
        self.assertTrue(np.all(target_c[:, 1] > pi0_c[:, 1]))
        for c in (0, 1):
            idx = self.clusters == c
            np.testing.assert_allclose(
                target[:, idx] / target_c[:, [c]],
                self.pi0[:, idx] / pi0_c[:, [c]],
                atol=1e-12,
            )

    def test_zero_target_strength_recovers_logger(self):
        target = build_cluster_favoring_target(
            self.q,
            self.pi0,
            self.clusters,
            tau=1.0,
            strength=0.0,
        )
        np.testing.assert_allclose(target, self.pi0, atol=1e-12)

    def test_behavior_overlap_degradation_preserves_rows_and_conditionals(self):
        target = build_cluster_favoring_target(
            self.q,
            self.pi0,
            self.clusters,
            tau=1.0,
            strength=1.0,
        )
        behavior_same = degrade_behavior_cluster_overlap(
            self.pi0,
            target,
            self.clusters,
            overlap_level=1.0,
        )
        np.testing.assert_allclose(behavior_same, self.pi0, atol=1e-12)

        behavior_low = degrade_behavior_cluster_overlap(
            self.pi0,
            target,
            self.clusters,
            overlap_level=0.0,
        )
        np.testing.assert_allclose(behavior_low.sum(axis=1), 1.0)
        behavior_c = cluster_policy_mass(behavior_low, self.clusters)
        target_c = cluster_policy_mass(target, self.clusters)
        favored = target_c > cluster_policy_mass(self.pi0, self.clusters) + 1e-12
        self.assertTrue(np.all(behavior_c[favored] == 0.0))

    def test_cluster_overlap_diagnostics_detects_support_failure(self):
        target = build_cluster_favoring_target(
            self.q,
            self.pi0,
            self.clusters,
            tau=1.0,
            strength=1.0,
        )
        behavior = degrade_behavior_cluster_overlap(
            self.pi0,
            target,
            self.clusters,
            overlap_level=0.0,
        )
        diagnostics = cluster_overlap_diagnostics(
            behavior,
            target,
            self.clusters,
            user_idx=np.array([0, 0, 1, 1]),
            observed_actions=np.array([0, 1, 0, 1]),
        )
        self.assertTrue(diagnostics["complete_support_failure"])
        self.assertGreater(diagnostics["unsupported_target_cluster_mass"], 0.0)
        self.assertGreaterEqual(
            diagnostics["clusters_with_contextual_zero_behavior_fraction"],
            diagnostics["clusters_without_any_behavior_support_fraction"],
        )

    def test_resample_logged_data_rebuilds_counts_and_means(self):
        base_data = {
            "n_rounds": 5,
            "n_users": 2,
            "n_actions": 4,
            "user_idx": np.array([0, 0, 0, 1, 1]),
            "fixed_user_contexts": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "fixed_expected_rewards": self.q,
            "action_context": np.arange(8).reshape(4, 2),
            "action_context_one_hot": np.eye(4),
            "cluster_indices": self.clusters,
        }
        behavior = np.zeros((2, 4))
        behavior[:, 0] = 1.0
        data = resample_logged_data(
            base_data,
            behavior,
            reward_std=0.0,
            sample_seed=123,
        )
        np.testing.assert_array_equal(data["action"], np.zeros(5, dtype=int))
        self.assertEqual(int(data["obs_count_mat"].sum()), 5)
        self.assertEqual(data["obs_count_mat"][0, 0], 3)
        self.assertEqual(data["obs_count_mat"][1, 0], 2)
        self.assertEqual(data["reward_mat"][0, 0], self.q[0, 0])
        self.assertEqual(data["reward_mat"][1, 0], self.q[1, 0])
        np.testing.assert_allclose(data["pscore"], np.ones(5))
        self.assertEqual(data["pi_b"].shape, (5, 4, 1))

    def test_policy_value_uses_population_target(self):
        target = build_cluster_favoring_target(
            self.q,
            self.pi0,
            self.clusters,
            tau=1.0,
            strength=0.0,
        )
        expected = float(np.sum(self.q * self.pi0, axis=1).mean())
        self.assertAlmostEqual(policy_value(self.q, target), expected)

    def test_aggregate_results_uses_seedwise_relative_mse(self):
        rows = []
        for seed, true_value in enumerate((10.0, 20.0)):
            rows.append(
                {
                    "seed": seed,
                    "partition": "matched",
                    "overlap_level": 0.5,
                    "target_tau": 1.0,
                    "target_strength": 1.0,
                    "estimate_offcem": true_value + 2.0,
                    "estimate_dr": true_value + 1.0,
                    "estimate_dm": true_value - 1.0,
                    "true_value": true_value,
                    "error_offcem": 2.0,
                    "error_dr": 1.0,
                    "error_dm": -1.0,
                    "sq_error_offcem": 4.0,
                    "sq_error_dr": 1.0,
                    "sq_error_dm": 1.0,
                    "cluster_weight_mean": 1.0,
                    "cluster_weight_max": 2.0,
                    "cluster_weight_variance": 0.5,
                    "cluster_ess": 8.0,
                    "cluster_ess_fraction": 0.8,
                    "population_cluster_weight_max": 3.0,
                    "population_cluster_weight_variance": 0.7,
                    "unsupported_target_cluster_mass": 0.0,
                    "clusters_without_any_behavior_support_fraction": 0.0,
                    "clusters_with_contextual_zero_behavior_fraction": 0.0,
                    "complete_support_failure": False,
                    "mean_target_favored_cluster_mass": 0.6,
                }
            )
        aggregate = aggregate_results(rows)[0]
        self.assertAlmostEqual(aggregate["mse_offcem"], 4.0)
        self.assertAlmostEqual(
            aggregate["rel_mse_offcem"],
            0.5 * ((4.0 / 100.0) + (4.0 / 400.0)),
        )
        self.assertAlmostEqual(aggregate["bias2_offcem"], 4.0)
        self.assertAlmostEqual(aggregate["variance_offcem"], 0.0)
        self.assertAlmostEqual(aggregate["offcem_mse_minus_dr"], 3.0)
        self.assertAlmostEqual(aggregate["complete_support_failure_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
