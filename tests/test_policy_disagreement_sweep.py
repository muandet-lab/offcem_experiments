import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from run_policy_disagreement_sweep import aggregate_results  # noqa: E402
from run_policy_disagreement_sweep import build_fixed_partitions  # noqa: E402
from run_policy_disagreement_sweep import build_policy_disagreement_target  # noqa: E402
from run_policy_disagreement_sweep import cluster_mass  # noqa: E402
from run_policy_disagreement_sweep import compute_local_correctness_diagnostics  # noqa: E402
from run_policy_disagreement_sweep import compute_policy_disagreement  # noqa: E402
from run_policy_disagreement_sweep import compute_within_cluster_chi2  # noqa: E402
from run_policy_disagreement_sweep import make_breakpoint_rows  # noqa: E402
from run_policy_disagreement_sweep import make_failure_map_rows  # noqa: E402
from run_policy_disagreement_sweep import parse_partitions  # noqa: E402
from run_policy_disagreement_sweep import policy_value  # noqa: E402
from run_policy_disagreement_sweep import within_cluster_policy_metrics  # noqa: E402
from analyze_policy_disagreement_breakpoints import make_boundary_rows  # noqa: E402
from analyze_policy_disagreement_breakpoints import summarize_cells  # noqa: E402
from analyze_policy_lc_alignment import correlation_rows  # noqa: E402
from analyze_policy_lc_alignment import final_lambda_rows  # noqa: E402
from analyze_policy_lc_alignment import summarize_alignment  # noqa: E402


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
        self.assertAlmostEqual(
            compute_within_cluster_chi2(self.pi0, pi_e, self.labels),
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
        chi2_mid = compute_within_cluster_chi2(self.pi0, pi_mid, self.labels)
        chi2_full = compute_within_cluster_chi2(self.pi0, pi_full, self.labels)
        self.assertGreater(d_mid, 0.0)
        self.assertGreater(d_full, d_mid)
        self.assertGreater(chi2_mid, 0.0)
        self.assertGreater(chi2_full, chi2_mid)

    def test_within_cluster_chi2_matches_conditional_ratio_variance(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=1.0,
            tau=1.0,
        )
        expected_by_user = np.zeros(self.q.shape[0])
        for c in (0, 1):
            mask = self.labels == c
            pi0_c = self.pi0[:, mask, 0].sum(axis=1)
            pie_c = pi_e[:, mask, 0].sum(axis=1)
            pi0_cond = self.pi0[:, mask, 0] / pi0_c[:, None]
            pie_cond = pi_e[:, mask, 0] / pie_c[:, None]
            ratio = pie_cond / pi0_cond
            ratio_mean = np.sum(pi0_cond * ratio, axis=1)
            ratio_var = np.sum(pi0_cond * (ratio - ratio_mean[:, None]) ** 2, axis=1)
            expected_by_user += pi0_c * ratio_var
        self.assertAlmostEqual(
            compute_within_cluster_chi2(self.pi0, pi_e, self.labels),
            float(expected_by_user.mean()),
        )

    def test_within_cluster_policy_metrics_include_pairwise_ratio_mse(self):
        pi_e = build_policy_disagreement_target(
            self.q,
            self.pi0,
            self.labels,
            lambda_=1.0,
            tau=1.0,
        )
        metrics = within_cluster_policy_metrics(self.pi0, pi_e, self.labels)
        self.assertGreater(metrics["within_cluster_tv"], 0.0)
        self.assertGreater(metrics["within_cluster_chi2"], 0.0)
        self.assertAlmostEqual(
            metrics["pairwise_ratio_mse"],
            2.0 * metrics["within_cluster_chi2"],
        )

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

    def test_parse_partitions_accepts_wfss_and_dps_aliases(self):
        self.assertEqual(
            parse_partitions(
                "matched,original,wfss,feature_bucket,dps,kmeans,random"
            ),
            ["matched", "wfss", "dps", "kmeans", "random"],
        )

    def test_build_fixed_partitions_supports_added_methods(self):
        bandit_data = {
            "action_context_one_hot": np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.2, 0.1],
                    [2.0, 2.0],
                    [2.1, 2.0],
                    [2.2, 2.1],
                ]
            ),
            "cluster_indices": np.array([0, 0, 0, 1, 1, 1]),
        }
        partitions = build_fixed_partitions(
            bandit_data,
            n_clusters=2,
            seed=123,
            partition_names=["matched", "wfss", "dps", "kmeans", "random"],
        )
        self.assertEqual(
            set(partitions),
            {"matched", "wfss", "dps", "kmeans", "random"},
        )
        for labels in partitions.values():
            self.assertEqual(labels.shape, (6,))

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
                "within_cluster_chi2": 0.0,
                "pairwise_ratio_mse": 0.0,
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
                "within_cluster_chi2": 0.0,
                "pairwise_ratio_mse": 0.0,
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
        self.assertAlmostEqual(aggregate["pairwise_ratio_mse_mean"], 0.0)
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
        self.assertTrue(aggregate["offcem_loses_to_dr"])
        self.assertTrue(aggregate["offcem_loses_to_dm"])
        self.assertAlmostEqual(aggregate["offcem_mse_minus_dr"], 3.0)
        self.assertAlmostEqual(aggregate["offcem_mse_minus_dm"], 3.0)

    def test_failure_map_and_breakpoint_outputs(self):
        aggregates = [
            {
                "partition": "matched",
                "lambda": 0.0,
                "tau": 1.0,
                "lc_dm_mse_pi0_mean": 1.0,
                "pairwise_lc_mse_mean": 2.0,
                "within_cluster_tv_mean": 0.0,
                "within_cluster_chi2_mean": 0.0,
                "pairwise_ratio_mse_mean": 0.0,
                "mse_offcem": 0.5,
                "mse_dr": 1.0,
                "mse_dm": 1.5,
                "rel_mse_offcem": 0.05,
                "rel_mse_dr": 0.1,
                "rel_mse_dm": 0.15,
                "offcem_loses_to_dr": False,
                "offcem_loses_to_dm": False,
                "offcem_mse_minus_dr": -0.5,
                "offcem_mse_minus_dm": -1.0,
                "ARI_to_generating_partition_mean": 1.0,
            },
            {
                "partition": "matched",
                "lambda": 0.5,
                "tau": 1.0,
                "lc_dm_mse_pi0_mean": 1.0,
                "pairwise_lc_mse_mean": 2.0,
                "within_cluster_tv_mean": 0.25,
                "within_cluster_chi2_mean": 0.5,
                "pairwise_ratio_mse_mean": 1.0,
                "mse_offcem": 2.0,
                "mse_dr": 1.0,
                "mse_dm": 1.5,
                "rel_mse_offcem": 0.2,
                "rel_mse_dr": 0.1,
                "rel_mse_dm": 0.15,
                "offcem_loses_to_dr": True,
                "offcem_loses_to_dm": True,
                "offcem_mse_minus_dr": 1.0,
                "offcem_mse_minus_dm": 0.5,
                "ARI_to_generating_partition_mean": 1.0,
            },
        ]
        failure_rows = make_failure_map_rows(aggregates)
        self.assertEqual(len(failure_rows), 2)
        self.assertEqual(failure_rows[0]["L_lc_dm_mse_pi0"], 1.0)
        self.assertEqual(failure_rows[1]["D_within_cluster_tv"], 0.25)
        self.assertEqual(failure_rows[1]["D_within_cluster_chi2"], 0.5)
        self.assertEqual(failure_rows[1]["D_pairwise_ratio_mse"], 1.0)

        breakpoint = make_breakpoint_rows(aggregates)[0]
        self.assertTrue(breakpoint["breakpoint_vs_dr_exists"])
        self.assertEqual(breakpoint["breakpoint_vs_dr_lambda"], 0.5)
        self.assertEqual(
            breakpoint["breakpoint_vs_dr_within_cluster_tv"],
            0.25,
        )
        self.assertEqual(
            breakpoint["breakpoint_vs_dr_within_cluster_chi2"],
            0.5,
        )
        self.assertEqual(
            breakpoint["breakpoint_vs_dr_pairwise_ratio_mse"],
            1.0,
        )
        self.assertTrue(breakpoint["breakpoint_vs_dm_exists"])

    def test_tolerance_breakpoint_uses_bootstrap_supported_material_loss(self):
        rows = []
        for seed in range(4):
            rows.append(
                {
                    "seed": seed,
                    "partition": "matched",
                    "lambda": 0.0,
                    "tau": 1.0,
                    "sq_error_offcem": 1.0,
                    "sq_error_dr": 1.0,
                    "within_cluster_tv": 0.0,
                    "within_cluster_chi2": 0.0,
                    "pairwise_ratio_mse": 0.0,
                    "lc_dm_mse_pi0": 2.0,
                    "pairwise_lc_mse": 4.0,
                    "theorem33_bias": 0.0,
                    "ARI_to_generating_partition": 1.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "partition": "matched",
                    "lambda": 0.5,
                    "tau": 1.0,
                    "sq_error_offcem": 2.0,
                    "sq_error_dr": 1.0,
                    "within_cluster_tv": 0.25,
                    "within_cluster_chi2": 0.5,
                    "pairwise_ratio_mse": 1.0,
                    "lc_dm_mse_pi0": 2.0,
                    "pairwise_lc_mse": 4.0,
                    "theorem33_bias": 1.0,
                    "ARI_to_generating_partition": 1.0,
                }
            )
        cells = summarize_cells(
            rows,
            delta=0.1,
            bootstrap_samples=200,
            alpha=0.05,
            random_state=123,
        )
        boundary = make_boundary_rows(cells)[0]
        self.assertTrue(boundary["breakpoint_ci_exists"])
        self.assertAlmostEqual(boundary["breakpoint_ci_lambda"], 0.05)
        self.assertAlmostEqual(boundary["breakpoint_ci_D_chi2"], 0.05)
        self.assertAlmostEqual(boundary["breakpoint_ci_pairwise_ratio_mse"], 0.1)

    def test_policy_lc_alignment_summary_uses_final_lambda(self):
        rows = []
        for lam in (0.0, 1.0):
            for seed in range(3):
                rows.append(
                    {
                        "seed": seed,
                        "partition": "matched",
                        "lambda": lam,
                        "tau": 1.0,
                        "error_offcem": lam + seed * 0.1,
                        "sq_error_offcem": 1.0 + lam,
                        "sq_error_dr": 1.0,
                        "within_cluster_tv": lam * 0.2,
                        "within_cluster_chi2": lam * 0.5,
                        "lc_dm_mse_pi0": 2.0,
                        "pairwise_lc_mse": 4.0,
                        "policy_lc_weighted_covariance": -lam,
                        "theorem33_bias": lam,
                    }
                )
        summaries = summarize_alignment(rows, delta=0.1)
        final = final_lambda_rows(summaries)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["lambda"], 1.0)
        self.assertAlmostEqual(
            final[0]["abs_policy_lc_weighted_covariance"],
            1.0,
        )
        self.assertTrue(final[0]["materially_worse_than_dr"])

    def test_policy_lc_alignment_correlations_are_emitted(self):
        rows = []
        for partition, lc in (("matched", 1.0), ("random", 3.0)):
            for lam in (0.0, 1.0):
                for seed in range(2):
                    rows.append(
                        {
                            "seed": seed,
                            "partition": partition,
                            "lambda": lam,
                            "tau": 1.0,
                            "error_offcem": lc * lam,
                            "sq_error_offcem": lc * lam + 1.0,
                            "sq_error_dr": 1.0,
                            "within_cluster_tv": lam,
                            "within_cluster_chi2": lam,
                            "lc_dm_mse_pi0": lc,
                            "pairwise_lc_mse": lc,
                            "policy_lc_weighted_covariance": lc * lam,
                            "theorem33_bias": -lc * lam,
                        }
                    )
        correlations = correlation_rows(summarize_alignment(rows, delta=0.1))
        self.assertTrue(correlations)
        self.assertIn("abs_policy_lc_weighted_covariance", {r["x"] for r in correlations})


if __name__ == "__main__":
    unittest.main()
