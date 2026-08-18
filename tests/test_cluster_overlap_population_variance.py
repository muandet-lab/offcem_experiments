import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from analyze_cluster_overlap_population_variance import score_moments  # noqa: E402


class ClusterOverlapPopulationVarianceTest(unittest.TestCase):
    def test_score_moments_match_manual_single_cluster_case(self):
        q = np.array([[1.0, 3.0]])
        f = np.array([[[0.5], [2.0]]])
        behavior = np.array([[0.25, 0.75]])
        target = np.array([[0.50, 0.50]])
        clusters = np.array([0, 0])

        moments = score_moments(
            q_population=q,
            f_hat=f,
            behavior_population=behavior,
            target_population=target,
            clusters=clusters,
            reward_std=2.0,
            n_rounds=10,
        )

        dm = 0.5 * 0.5 + 0.5 * 2.0
        residual = q[0] - f[0, :, 0]
        mean = dm + np.sum(behavior[0] * residual)
        second = (
            dm**2
            + 2.0 * dm * np.sum(behavior[0] * residual)
            + np.sum(behavior[0] * (residual**2 + 4.0))
        )
        variance = second - mean**2

        self.assertAlmostEqual(moments["exact_score_mean"], mean)
        self.assertAlmostEqual(moments["exact_score_variance"], variance)
        self.assertAlmostEqual(moments["exact_estimator_variance"], variance / 10)
        self.assertFalse(moments["exact_support_failure"])

    def test_score_moments_flags_cluster_support_failure(self):
        q = np.array([[1.0, 3.0]])
        f = np.array([[1.0, 3.0]])
        behavior = np.array([[1.0, 0.0]])
        target = np.array([[0.5, 0.5]])
        clusters = np.array([0, 1])

        moments = score_moments(
            q_population=q,
            f_hat=f,
            behavior_population=behavior,
            target_population=target,
            clusters=clusters,
            reward_std=1.0,
            n_rounds=5,
        )

        self.assertTrue(moments["exact_support_failure"])
        self.assertAlmostEqual(
            moments["exact_unsupported_target_cluster_mass"],
            0.5,
        )
        self.assertTrue(np.isinf(moments["exact_population_cluster_weight_max"]))


if __name__ == "__main__":
    unittest.main()
