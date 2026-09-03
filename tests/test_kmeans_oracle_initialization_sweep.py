import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "synthetic"))

from run_kmeans_oracle_initialization_sweep import (
    DEFAULT_K_BEST_GRID,
    build_partitions,
    k_cases_for_n,
    oracle_initial_centers,
    parse_k_best_grid,
    proportional_subcluster_counts,
)
from run_sample_size_stress_test import parse_n_list


class KMeansOracleInitializationSweepTest(unittest.TestCase):
    def setUp(self):
        self.features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [10.0, 10.0],
                [10.0, 11.0],
                [11.0, 10.0],
            ]
        )
        self.labels = np.asarray([0, 0, 0, 1, 1, 1])

    def test_default_grid_and_cases_match_the_preselected_values(self):
        n_list = parse_n_list("500,1000,3000,10000,30000,100000")
        selected = parse_k_best_grid(DEFAULT_K_BEST_GRID, n_list, n_actions=1000)
        self.assertEqual(selected[1000], 30)
        self.assertEqual(selected[100000], 450)
        self.assertEqual(k_cases_for_n(selected[500]), [50])
        self.assertEqual(k_cases_for_n(selected[10000]), [50, 150])

    def test_exact_generating_centroids_are_used_when_k_matches(self):
        centers, initialization_type = oracle_initial_centers(
            self.features, self.labels, n_clusters=2, random_state=3
        )
        np.testing.assert_allclose(centers, [[1 / 3, 1 / 3], [31 / 3, 31 / 3]])
        self.assertEqual(initialization_type, "generating_centroids")

    def test_oracle_centers_can_coarsen_and_split_the_generating_partition(self):
        coarsened, coarsened_type = oracle_initial_centers(
            self.features, self.labels, n_clusters=1, random_state=3
        )
        split, split_type = oracle_initial_centers(
            self.features, self.labels, n_clusters=4, random_state=3
        )
        self.assertEqual(coarsened.shape, (1, 2))
        self.assertEqual(split.shape, (4, 2))
        self.assertEqual(coarsened_type, "oracle_coarsened_generating_centroids")
        self.assertEqual(split_type, "oracle_split_generating_centroids")

    def test_proportional_split_allocation_is_feasible(self):
        counts = np.asarray([2, 3, 5])
        allocation = proportional_subcluster_counts(counts, n_centers=7)
        self.assertEqual(allocation.sum(), 7)
        self.assertTrue(np.all(allocation >= 1))
        self.assertTrue(np.all(allocation <= counts))

    def test_both_partition_types_are_constructed_for_each_k(self):
        world = {
            "action_context_one_hot": self.features,
            "cluster_indices": self.labels,
        }
        partitions = build_partitions(world, requested_k=[2, 4], partition_seed=21)
        self.assertEqual(set(partitions), {(2, "standard"), (2, "oracle_initialized"), (4, "standard"), (4, "oracle_initialized")})
        self.assertTrue(np.isfinite(partitions[(2, "oracle_initialized")]["initial_ari_to_generating"]))


if __name__ == "__main__":
    unittest.main()
