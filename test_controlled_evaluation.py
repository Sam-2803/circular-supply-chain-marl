import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from controlled_evaluation import run_controlled_evaluation
from marl_agents import AGENT_NAMES, LinearDQNAgent


def make_checkpoints(directory):
    for index, name in enumerate(AGENT_NAMES):
        agent = LinearDQNAgent(name, observation_size=7, seed=index)
        agent.save(Path(directory) / f"{name}.npz")


class TestControlledEvaluation(unittest.TestCase):

    def test_horizon_seed_alignment_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "models"
            output_dir = Path(directory) / "results"
            make_checkpoints(checkpoint_dir)
            detailed, summary, alignment = run_controlled_evaluation(
                [10, 11], checkpoint_dir, output_dir
            )
            self.assertEqual(set(detailed["episode_length"]), {20})
            self.assertEqual(set(detailed["policy"]), {
                "no_op", "random", "heuristic", "trained"
            })
            self.assertEqual(len(detailed), 8)
            self.assertTrue(alignment[[
                "demand_equal_to_no_op",
                "yield_equal_to_no_op",
                "transport_equal_to_no_op",
            ]].to_numpy().all())
            self.assertIn("cost_mean", summary.columns)
            self.assertIn("reward_mean_mean", summary.columns)
            self.assertTrue((output_dir / "evaluation_per_seed.csv").exists())
            self.assertTrue((output_dir / "evaluation_summary.csv").exists())
            self.assertTrue((output_dir / "evaluation_metadata.json").exists())

    def test_repeated_controlled_runs_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "models"
            make_checkpoints(checkpoint_dir)
            first, _, _ = run_controlled_evaluation(
                [20, 21], checkpoint_dir, Path(directory) / "first"
            )
            second, _, _ = run_controlled_evaluation(
                [20, 21], checkpoint_dir, Path(directory) / "second"
            )
            pd.testing.assert_frame_equal(first, second)

    def test_trained_evaluation_does_not_update_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "models"
            make_checkpoints(checkpoint_dir)
            before = {
                name: np.load(checkpoint_dir / f"{name}.npz")["weights"].copy()
                for name in AGENT_NAMES
            }
            run_controlled_evaluation(
                [30], checkpoint_dir, Path(directory) / "results"
            )
            for name in AGENT_NAMES:
                after = np.load(checkpoint_dir / f"{name}.npz")["weights"]
                np.testing.assert_array_equal(before[name], after)

    def test_proxy_and_unavailable_metrics_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "models"
            make_checkpoints(checkpoint_dir)
            detailed, _, _ = run_controlled_evaluation(
                [40], checkpoint_dir, Path(directory) / "results"
            )
            self.assertIn("cost_per_delivered_unit", detailed.columns)
            self.assertIn("wip", detailed.columns)
            self.assertTrue(detailed["wip"].isna().all())


if __name__ == "__main__":
    unittest.main()