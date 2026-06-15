import sys
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SYNTHETIC = Path(__file__).resolve().parents[1] / "src" / "synthetic"
sys.path.insert(0, str(SYNTHETIC))

from applicability.analysis import _benjamini_hochberg  # noqa: E402
from applicability.analysis import _load_records  # noqa: E402
from applicability.config import build_cells  # noqa: E402
from applicability.config import config_hash  # noqa: E402
from applicability.config import task_coordinates  # noqa: E402
from applicability.config import task_index  # noqa: E402
from applicability.config import write_manifest  # noqa: E402
from applicability.dataset import ApplicabilityDataset  # noqa: E402
from applicability.diagnostics import demeaned_error  # noqa: E402
from applicability.runner import checkpoint_summary  # noqa: E402
from applicability.runner import run_task  # noqa: E402


class ApplicabilityConfigTest(unittest.TestCase):
    def test_explicit_design_merges_nested_defaults(self):
        config = {
            "defaults": {
                "n_rounds": 100,
                "reward": {"alpha": [1, 0, 0, 0], "noise_std": 1},
            },
            "design": {
                "type": "explicit",
                "points": [{"reward": {"noise_std": 3}}],
            },
        }
        cells = build_cells(config)
        self.assertEqual(cells[0]["reward"]["alpha"], [1, 0, 0, 0])
        self.assertEqual(cells[0]["reward"]["noise_std"], 3)
        self.assertEqual(config_hash(cells[0]), config_hash(cells[0]))

    def test_sobol_design_is_reproducible(self):
        config = {
            "defaults": {"reward": {"alpha": [0.25] * 4}},
            "design": {
                "type": "sobol",
                "n_cells": 4,
                "seed": 7,
                "sample_reward_simplex": True,
                "factors": {"n_rounds": {"values": [100, 200]}},
            },
        }
        first = build_cells(config)
        second = build_cells(config)
        self.assertEqual(first, second)
        for cell in first:
            self.assertAlmostEqual(sum(cell["reward"]["alpha"]), 1.0)

    def test_task_index_round_trip(self):
        for cell_index in range(3):
            for seed_index in range(5):
                index = task_index(cell_index, seed_index, 5)
                self.assertEqual(
                    task_coordinates(index, n_cells=3, n_seeds=5),
                    (cell_index, seed_index),
                )

    def test_task_index_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            task_coordinates(15, n_cells=3, n_seeds=5)

    def test_manifest_contains_linear_task_mapping(self):
        cells = [{"name": "a"}, {"name": "b"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_manifest(
                "config.yaml",
                {"defaults": {}},
                cells,
                output,
                n_seeds=2,
            )
            with open(output / "manifest.json") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["n_tasks"], 4)
            self.assertEqual(
                [
                    (item["task_index"], item["cell_index"], item["seed_index"])
                    for item in manifest["tasks"]
                ],
                [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)],
            )


class ApplicabilityDatasetTest(unittest.TestCase):
    def _dataset(self, deficient_fraction):
        return ApplicabilityDataset(
            n_actions=20,
            n_users=10,
            dim_context=4,
            n_cat_dim=4,
            n_cat_per_dim=3,
            n_unobserved_cat_dim=1,
            n_clusters=4,
            gen_clustering_method="feature_bucket",
            cluster_balance="natural",
            cluster_temperature=10.0,
            alpha=[0.4, 0.3, 0.2, 0.1],
            feature_nonlinearity=0.5,
            reward_scale=2.0,
            reward_std=1.0,
            beta=-0.1,
            target_epsilon=0.2,
            deficient_action_fraction=deficient_fraction,
            random_state=123,
        )

    def test_support_changes_do_not_change_reward_or_target(self):
        full = self._dataset(0.0).generate(100, sample_random_state=500)
        deficient = self._dataset(0.3).generate(
            100, sample_random_state=500
        )
        np.testing.assert_allclose(
            full["fixed_expected_rewards"],
            deficient["fixed_expected_rewards"],
        )
        np.testing.assert_allclose(
            full["target_policy_population"],
            deficient["target_policy_population"],
        )
        self.assertEqual(full["true_policy_value"], deficient["true_policy_value"])
        self.assertLess(deficient["support_mask"].sum(), full["support_mask"].sum())

    def test_sample_seed_changes_log_not_environment(self):
        first = self._dataset(0.0).generate(100, sample_random_state=500)
        second = self._dataset(0.0).generate(100, sample_random_state=501)
        np.testing.assert_allclose(
            first["fixed_expected_rewards"], second["fixed_expected_rewards"]
        )
        self.assertFalse(np.array_equal(first["action"], second["action"]))
        self.assertAlmostEqual(
            sum(first["realized_variance_shares"].values()), 1.0
        )


class ApplicabilityDiagnosticsTest(unittest.TestCase):
    def test_demeaning_removes_cluster_offsets(self):
        truth = np.array([[1.0, 2.0, 4.0, 8.0]])
        prediction = truth + np.array([[10.0, 10.0, -3.0, -3.0]])
        clusters = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(
            demeaned_error(prediction, truth, clusters), 0.0, atol=1e-12
        )

    def test_bh_adjustment_is_monotone_in_rank(self):
        p_values = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted = _benjamini_hochberg(p_values)
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))
        self.assertTrue(np.all(adjusted >= p_values))


class ApplicabilityRunnerTest(unittest.TestCase):
    def test_task_checkpoint_is_cached_and_summarized(self):
        cell = {"name": "test"}
        record = {
            "cell_id": config_hash(cell),
            "seed_index": 0,
            "partitions": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "applicability.runner.run_cell_seed",
                return_value=record,
            ) as mocked:
                first = run_task(
                    cells=[cell],
                    output_dir=directory,
                    n_seeds=1,
                    base_sample_seed=10,
                    phase="test",
                    cell_index=0,
                    seed_index=0,
                )
                second = run_task(
                    cells=[cell],
                    output_dir=directory,
                    n_seeds=1,
                    base_sample_seed=10,
                    phase="test",
                    cell_index=0,
                    seed_index=0,
                )
            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "ok")
            self.assertEqual(mocked.call_count, 1)
            summary = checkpoint_summary([cell], directory, 1)
            self.assertEqual(summary["ok"], 1)
            self.assertEqual(summary["missing"], 0)
            self.assertEqual(summary["error"], 0)

    def test_partition_failure_marks_task_failed(self):
        cell = {"name": "test"}
        record = {
            "cell_id": config_hash(cell),
            "seed_index": 0,
            "partitions": [{"label": "bad", "error": "failed"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "applicability.runner.run_cell_seed",
                return_value=record,
            ):
                result = run_task(
                    cells=[cell],
                    output_dir=directory,
                    n_seeds=1,
                    base_sample_seed=10,
                    phase="test",
                    cell_index=0,
                    seed_index=0,
                )
            self.assertEqual(result["status"], "error")
            summary = checkpoint_summary([cell], directory, 1)
            self.assertEqual(summary["error"], 1)
            path = next((Path(directory) / "checkpoints").glob("*.json"))
            with open(path) as file:
                persisted = json.load(file)
            self.assertEqual(persisted["status"], "error")

    def test_analysis_loader_ignores_stale_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "checkpoints"
            checkpoints.mkdir()
            for name in ("expected.json", "stale.json"):
                with open(checkpoints / name, "w") as file:
                    json.dump({"status": "ok", "name": name}, file)
            records = _load_records(
                Path(directory),
                expected_checkpoints={"expected.json"},
            )
            self.assertEqual([record["name"] for record in records], ["expected.json"])


if __name__ == "__main__":
    unittest.main()
