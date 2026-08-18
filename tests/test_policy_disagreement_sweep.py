import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from run_policy_disagreement_sweep import aggregate_results  # noqa: E402
from run_policy_disagreement_sweep import build_policy_disagreement_target  # noqa: E402
from run_policy_disagreement_sweep import cluster_mass  # noqa: E402
from run_policy_disagreement_sweep import compute_local_correctness_diagnostics  # noqa: E402
from run_policy_disagreement_sweep import compute_policy_disagreement  # noqa: E402
from run_policy_disagreement_sweep import policy_value  # noqa: E402


class PolicyDisagreementSweepTest(unittest.TestCase):
    def setUp(self):
        self.q = np.array(
            [
                [0.0, 2.0, 1.0, 3.0],
                [1.0, 0.0, 4.0, 2.0],
            ]
        )
        self.pi0 = np.array(
            [
                [0.20, 0.30, 0.10, 0.40],
                [0.15, 0.35, 0.25, 0.25],
            ]
        )[:, :, np.newaxis]
        self.labels = np.array([0, 0, 1, 1])

    def test_target_policy_rows_sum_to_one(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=0.75,
            tau=1.0,
        )
        np.testing.assert_allclose(pi_e[:, :, 0].sum(axis=1), 1.0)
        self.assertTrue(np.all(pi_e >= 0.0))

    def test_cluster_mass_is_preserved(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=1.0,
            tau=1.0,
        )
        np.testing.assert_allclose(
            cluster_mass(pi_e, self.labels),
            cluster_mass(self.pi0, self.labels),
            atol=1e-12,
        )

    def test_lambda_zero_recovers_logger_and_zero_disagreement(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=0.0,
            tau=1.0,
        )
        np.testing.assert_allclose(pi_e, self.pi0, atol=1e-12)
        self.assertAlmostEqual(
            compute_policy_disagreement(self.pi0, pi_e, self.labels),
            0.0,
        )

    def test_policy_disagreement_increases_with_lambda(self):
        pi_mid = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=0.5,
            tau=1.0,
        )
        pi_full = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=1.0,
            tau=1.0,
        )
        d_mid = compute_policy_disagreement(self.pi0, pi_mid, self.labels)
        d_full = compute_policy_disagreement(self.pi0, pi_full, self.labels)
        self.assertGreater(d_mid, 0.0)
        self.assertGreater(d_full, d_mid)

    def test_policy_value_uses_population_q(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=0.0,
            tau=1.0,
        )
        expected = float(np.sum(self.q * self.pi0[:, :, 0], axis=1).mean())
        self.assertAlmostEqual(policy_value(self.q, pi_e), expected)

    def test_local_correctness_diagnostics(self):
        f_hat = (self.q + np.array(
            [
                [1.0, 3.0, -2.0, 0.0],
                [2.0, 0.0, 1.0, -1.0],
            ]
        ))[:, :, np.newaxis]
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=1.0,
            tau=1.0,
        )
        diagnostics = compute_local_correctness_diagnostics(
            self.q,
            f_hat,
            self.pi0,
            pi_e,
            self.labels,
        )
        error = f_hat[:, :, 0] - self.q
        expected_bias_proxy = float(
            np.sum((self.pi0[:, :, 0] - pi_e[:, :, 0]) * error, axis=1).mean()
        )
        self.assertGreater(diagnostics["reward_mse"], 0.0)
        self.assertGreater(diagnostics["lc_dm_mse_uniform"], 0.0)
        self.assertGreater(diagnostics["lc_dm_mse_pi0"], 0.0)
        self.assertGreater(diagnostics["pairwise_lc_mse"], 0.0)
        self.assertGreater(
            diagnostics["within_cluster_ratio_pairwise_mse"],
            0.0,
        )
        self.assertAlmostEqual(
            diagnostics["population_bias_formula"],
            expected_bias_proxy,
        )
        self.assertAlmostEqual(
            diagnostics["theorem33_bias"],
            -expected_bias_proxy,
        )
        self.assertAlmostEqual(
            diagnostics["policy_lc_weighted_covariance"],
            expected_bias_proxy,
        )

    def test_bias_proxy_is_zero_when_lambda_zero(self):
        f_hat = (self.q + np.array(
            [
                [1.0, 3.0, -2.0, 0.0],
                [2.0, 0.0, 1.0, -1.0],
            ]
        ))[:, :, np.newaxis]
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=0.0,
            tau=1.0,
        )
        diagnostics = compute_local_correctness_diagnostics(
            self.q,
            f_hat,
            self.pi0,
            pi_e,
            self.labels,
        )
        self.assertAlmostEqual(diagnostics["population_bias_formula"], 0.0)
        self.assertAlmostEqual(diagnostics["theorem33_bias"], 0.0)
        self.assertAlmostEqual(
            diagnostics["policy_lc_weighted_covariance"],
            0.0,
        )
        self.assertAlmostEqual(
            diagnostics["within_cluster_ratio_pairwise_mse"],
            0.0,
        )

    def test_rel_mse_is_seedwise_normalized(self):
        rows = [
            {
                "seed": 0,
                "partition": "matched",
                "lambda": 0.0,
                "tau": 1.0,
                "estimate_offcem": 12.0,
                "estimate_dr": 11.0,
                "estimate_dm": 9.0,
                "true_value": 10.0,
                "error_offcem": 2.0,
                "error_dr": 1.0,
                "error_dm": -1.0,
                "sq_error_offcem": 4.0,
                "sq_error_dr": 1.0,
                "sq_error_dm": 1.0,
                "within_cluster_tv": 0.0,
                "within_cluster_ratio_pairwise_mse": 0.0,
                "cluster_weight_max_abs_dev_from_1": 1e-12,
                "reward_mse": 0.5,
                "lc_dm_mse_uniform": 1.0,
                "lc_dm_mse_pi0": 2.0,
                "pairwise_lc_mse": 3.0,
                "population_bias_formula": 0.1,
                "theorem33_bias": -0.1,
                "policy_lc_weighted_covariance": 0.1,
                "ARI_to_generating_partition": 1.0,
            },
            {
                "seed": 1,
                "partition": "matched",
                "lambda": 0.0,
                "tau": 1.0,
                "estimate_offcem": 22.0,
                "estimate_dr": 21.0,
                "estimate_dm": 19.0,
                "true_value": 20.0,
                "error_offcem": 2.0,
                "error_dr": 1.0,
                "error_dm": -1.0,
                "sq_error_offcem": 4.0,
                "sq_error_dr": 1.0,
                "sq_error_dm": 1.0,
                "within_cluster_tv": 0.0,
                "within_cluster_ratio_pairwise_mse": 0.0,
                "cluster_weight_max_abs_dev_from_1": 2e-12,
                "reward_mse": 1.5,
                "lc_dm_mse_uniform": 3.0,
                "lc_dm_mse_pi0": 4.0,
                "pairwise_lc_mse": 5.0,
                "population_bias_formula": 0.3,
                "theorem33_bias": -0.3,
                "policy_lc_weighted_covariance": 0.3,
                "ARI_to_generating_partition": 1.0,
            },
        ]
        aggregate = aggregate_results(rows)[0]
        self.assertAlmostEqual(aggregate["mse_offcem"], 4.0)
        self.assertAlmostEqual(aggregate["rel_mse_offcem"], 0.5 * (0.04 + 0.01))
        self.assertAlmostEqual(aggregate["bias2_offcem"], 4.0)
        self.assertAlmostEqual(aggregate["variance_offcem"], 0.0)
        self.assertAlmostEqual(aggregate["mse_dr"], 1.0)
        self.assertAlmostEqual(aggregate["mse_dm"], 1.0)
        self.assertAlmostEqual(aggregate["cluster_weight_max_abs_dev_from_1_max"], 2e-12)
        self.assertAlmostEqual(aggregate["reward_mse_mean"], 1.0)
        self.assertAlmostEqual(aggregate["lc_dm_mse_uniform_mean"], 2.0)
        self.assertAlmostEqual(aggregate["lc_dm_mse_pi0_mean"], 3.0)
        self.assertAlmostEqual(aggregate["pairwise_lc_mse_mean"], 4.0)
        self.assertAlmostEqual(aggregate["population_bias_formula_mean"], 0.2)
        self.assertAlmostEqual(aggregate["theorem33_bias_mean"], -0.2)
        self.assertAlmostEqual(
            aggregate["policy_lc_weighted_covariance_mean"],
            0.2,
        )
        self.assertAlmostEqual(aggregate["ARI_to_generating_partition_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
