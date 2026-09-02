import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "synthetic"))

from run_kmeans_granularity_sweep import DEFAULT_K_GRID
from run_kmeans_granularity_sweep import aggregate_results
from run_kmeans_granularity_sweep import parse_kmeans_k_grid
from run_sample_size_stress_test import parse_n_list


class KMeansGranularitySweepTest(unittest.TestCase):
    def test_default_grid_matches_requested_round_values(self):
        n_list = parse_n_list("500,1000,3000,10000,30000,100000")
        grid = parse_kmeans_k_grid(DEFAULT_K_GRID, n_list, n_actions=1000)
        self.assertEqual(grid[500], [10, 30, 50])
        self.assertEqual(grid[1000], [10, 30, 50])
        self.assertEqual(grid[3000], [30, 50, 70])
        self.assertEqual(grid[10000], [50, 100, 150])
        self.assertEqual(grid[30000], [50, 100, 150, 200])
        self.assertEqual(grid[100000], [50, 250, 350, 450])

    def test_grid_requires_every_requested_sample_size(self):
        with self.assertRaises(ValueError):
            parse_kmeans_k_grid("500:10,30", [500, 1000], n_actions=1000)

    def test_aggregate_preserves_tv_metric(self):
        rows = [
            {
                "n": 500,
                "kmeans_k": 10,
                "world_seed": 1,
                "estimate": 3.0,
                "true_value": 2.0,
                "error": 1.0,
                "within_cluster_tv": 0.2,
                "within_cluster_chi2": 0.3,
                "pairwise_ratio_mse": 0.6,
                "reward_mse": 1.0,
                "lc_error": 2.0,
                "ari_to_matched": 0.1,
                "cluster_size_min": 2.0,
                "cluster_size_max": 20.0,
                "cluster_size_std": 3.0,
                "cluster_ess_fraction": 0.4,
                "cluster_weight_max": 5.0,
                "cluster_weight_variance": 6.0,
                "pairwise_training_examples": 7.0,
                "target_mass_on_observed_actions": 0.8,
                "user_action_coverage": 0.01,
            },
            {
                "n": 500,
                "kmeans_k": 10,
                "world_seed": 2,
                "estimate": 1.0,
                "true_value": 2.0,
                "error": -1.0,
                "within_cluster_tv": 0.4,
                "within_cluster_chi2": 0.5,
                "pairwise_ratio_mse": 1.0,
                "reward_mse": 1.0,
                "lc_error": 2.0,
                "ari_to_matched": 0.1,
                "cluster_size_min": 2.0,
                "cluster_size_max": 20.0,
                "cluster_size_std": 3.0,
                "cluster_ess_fraction": 0.4,
                "cluster_weight_max": 5.0,
                "cluster_weight_variance": 6.0,
                "pairwise_training_examples": 7.0,
                "target_mass_on_observed_actions": 0.8,
                "user_action_coverage": 0.01,
            },
        ]
        aggregate = aggregate_results(rows)[0]
        self.assertAlmostEqual(aggregate["within_cluster_tv_mean"], 0.3)
        self.assertEqual(aggregate["mse"], 1.0)
        self.assertEqual(aggregate["bias2"], 0.0)


if __name__ == "__main__":
    unittest.main()
