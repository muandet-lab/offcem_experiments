import sys
from pathlib import Path
import unittest

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from clustering import compute_clusters  # noqa: E402
from run_sample_size_stress_test import build_feedback_prefix  # noqa: E402
from run_sample_size_stress_test import build_fixed_estimation_partitions  # noqa: E402
from run_sample_size_stress_test import build_partition_sweep  # noqa: E402
from run_sample_size_stress_test import aggregate_results  # noqa: E402
from run_sample_size_stress_test import generate_world  # noqa: E402
from run_sample_size_stress_test import plot_errorbars_by_group  # noqa: E402
from run_sample_size_stress_test import population_policy_value  # noqa: E402
from run_sample_size_stress_test import sample_logged_stream  # noqa: E402


class SampleSizeStressPrefixTest(unittest.TestCase):
    def _full_bandit_data(self):
        fixed_expected_rewards = np.array(
            [
                [1.0, 2.0, 0.0],
                [0.5, 1.5, 3.0],
            ]
        )
        user_idx = np.array([0, 0, 1, 0, 1, 0])
        action = np.array([1, 1, 2, 0, 2, 1])
        reward = np.array([10.0, 14.0, 4.0, 2.0, 8.0, 16.0])
        expected_reward = fixed_expected_rewards[user_idx]
        pi_b = np.full((6, 3, 1), 1.0 / 3.0)
        return {
            "n_rounds": 6,
            "n_users": 2,
            "n_actions": 3,
            "clusters": np.eye(2)[np.array([0, 0, 1])][None, :, :].repeat(2, axis=0),
            "cluster_indices": np.array([0, 0, 1]),
            "action_context": np.array([[0], [1], [2]]),
            "action_context_one_hot": np.eye(3),
            "fixed_user_contexts": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "fixed_expected_rewards": fixed_expected_rewards,
            "g_x_e": np.zeros((2, 2)),
            "p_e_a": np.ones((3, 1, 1)),
            "position": None,
            "user_idx": user_idx,
            "context": np.array([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.3, 0.4], [0.1, 0.2]]),
            "action": action,
            "reward": reward,
            "expected_reward": expected_reward,
            "pi_b": pi_b,
            "pscore": pi_b[np.arange(6), action, 0],
            "action_embed": np.array([[1], [1], [2], [0], [2], [1]]),
        }

    def test_build_feedback_prefix_slices_round_fields(self):
        full = self._full_bandit_data()
        prefix = build_feedback_prefix(full, 4)
        self.assertEqual(prefix["n_rounds"], 4)
        for key in (
            "user_idx",
            "context",
            "action",
            "reward",
            "expected_reward",
            "pi_b",
            "pscore",
            "action_embed",
        ):
            self.assertEqual(prefix[key].shape[0], 4)
            np.testing.assert_array_equal(prefix[key], full[key][:4])

    def test_reward_averaging_and_counts_for_repeated_observations(self):
        prefix = build_feedback_prefix(self._full_bandit_data(), 6)
        self.assertEqual(int(prefix["obs_count_mat"].sum()), 6)
        self.assertEqual(prefix["obs_count_mat"][0, 1], 3)
        self.assertAlmostEqual(prefix["reward_sum_mat"][0, 1], 40.0)
        self.assertAlmostEqual(prefix["reward_mat"][0, 1], 40.0 / 3.0)
        self.assertEqual(prefix["obs_mat"][0, 1], 1)
        self.assertEqual(prefix["obs_mat"][1, 0], 0)

    def test_fixed_v_true_uses_population_not_prefix_rows(self):
        full = self._full_bandit_data()
        true_value, pi_e_population = population_policy_value(
            full["fixed_expected_rewards"],
            eps=0.2,
        )
        prefix = build_feedback_prefix(full, 3)
        repeated_prefix = build_feedback_prefix(full, 6)
        self.assertAlmostEqual(
            true_value,
            population_policy_value(prefix["fixed_expected_rewards"], eps=0.2)[0],
        )
        self.assertAlmostEqual(
            true_value,
            population_policy_value(repeated_prefix["fixed_expected_rewards"], eps=0.2)[0],
        )
        np.testing.assert_array_equal(
            pi_e_population[prefix["user_idx"]],
            pi_e_population[full["user_idx"][:3]],
        )

    def test_nested_prefix_equality(self):
        full = self._full_bandit_data()
        p3 = build_feedback_prefix(full, 3)
        p6 = build_feedback_prefix(full, 6)
        for key in (
            "user_idx",
            "context",
            "action",
            "reward",
            "expected_reward",
            "pi_b",
            "pscore",
            "action_embed",
        ):
            np.testing.assert_array_equal(p6[key][:3], p3[key])

    def test_cluster_partitions_are_independent_of_prefix_size(self):
        full = self._full_bandit_data()
        p3 = build_feedback_prefix(full, 3)
        p6 = build_feedback_prefix(full, 6)
        np.testing.assert_array_equal(p3["cluster_indices"], p6["cluster_indices"])

        wfss_a = compute_clusters(
            full["action_context_one_hot"],
            n_clusters=2,
            method="original",
            random_state=123,
        )
        wfss_b = compute_clusters(
            p3["action_context_one_hot"],
            n_clusters=2,
            method="original",
            random_state=123,
        )
        np.testing.assert_array_equal(wfss_a, wfss_b)

    def test_fixed_estimation_partitions_are_built_once_from_world_features(self):
        full = self._full_bandit_data()
        partitions = build_fixed_estimation_partitions(
            full_bandit_data=full,
            n_clusters=2,
            seed=12345,
        )
        self.assertEqual(
            set(partitions), {"matched", "reestimated_wfss", "kmeans"}
        )
        for labels in partitions.values():
            self.assertEqual(labels.shape, (3,))

    def test_plot_errorbars_are_bootstrapped_by_metric(self):
        rows = [
            {
                "n": 500,
                "seed": 0,
                "estimator": "DR",
                "estimate": 11.0,
                "true_value": 10.0,
                "squared_error": 1.0,
            },
            {
                "n": 500,
                "seed": 1,
                "estimator": "DR",
                "estimate": 14.0,
                "true_value": 10.0,
                "squared_error": 16.0,
            },
            {
                "n": 1000,
                "seed": 0,
                "estimator": "DR",
                "estimate": 12.0,
                "true_value": 10.0,
                "squared_error": 4.0,
            },
        ]
        errorbars = plot_errorbars_by_group(rows)
        self.assertIn((500, "DR"), errorbars)
        self.assertGreater(errorbars[(500, "DR")]["rel_mse"], 0.0)
        self.assertGreater(errorbars[(500, "DR")]["bias2"], 0.0)
        self.assertEqual(errorbars[(1000, "DR")]["rel_mse"], 0.0)

    def test_fixed_world_is_independent_of_stream_length(self):
        world = generate_world(
            world_seed=17,
            n_clusters=2,
            eps=0.2,
            reward_std=3.0,
        )
        short = sample_logged_stream(world, 3, stream_seed=91, reward_std=3.0)
        long = sample_logged_stream(world, 6, stream_seed=91, reward_std=3.0)
        np.testing.assert_array_equal(
            short["fixed_expected_rewards"], long["fixed_expected_rewards"]
        )
        np.testing.assert_array_equal(
            short["cluster_indices"], long["cluster_indices"]
        )
        for key in ("user_idx", "action", "reward"):
            np.testing.assert_array_equal(short[key], long[key][:3])

    def test_partition_sweep_has_deterministic_corruption_levels(self):
        world = generate_world(
            world_seed=18,
            n_clusters=2,
            eps=0.2,
            reward_std=3.0,
        )
        first = build_partition_sweep(
            world, 2, 100, ["matched", "corrupt"], [0.0, 0.5]
        )
        second = build_partition_sweep(
            world, 2, 100, ["matched", "corrupt"], [0.0, 0.5]
        )
        self.assertEqual(set(first), {"matched", "matched_corrupt_0", "matched_corrupt_0.5"})
        np.testing.assert_array_equal(first["matched"]["labels"], world["cluster_indices"])
        for key in first:
            np.testing.assert_array_equal(first[key]["labels"], second[key]["labels"])

    def test_hierarchical_aggregation_averages_within_world_first(self):
        rows = []
        for world_seed, errors in ((1, (1.0, 3.0)), (2, (-2.0, -4.0))):
            for stream_index, error in enumerate(errors):
                rows.append(
                    {
                        "n": 500,
                        "seed": world_seed,
                        "world_seed": world_seed,
                        "stream_seed": 100 + stream_index,
                        "arm": "end_to_end",
                        "partition": "matched",
                        "estimator": "OffCEM matched",
                        "estimate": 10.0 + error,
                        "true_value": 10.0,
                        "squared_error": error**2,
                    }
                )
        aggregate = aggregate_results(rows)[0]
        self.assertAlmostEqual(aggregate["mse"], 7.5)
        self.assertAlmostEqual(aggregate["bias2"], 6.5)
        self.assertAlmostEqual(aggregate["variance"], 1.0)
        self.assertAlmostEqual(aggregate["rel_mse"], 0.075)


if __name__ == "__main__":
    unittest.main()
