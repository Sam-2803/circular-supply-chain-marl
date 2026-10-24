import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from marl_agents import AGENT_NAMES, LinearDQNAgent
from reward_diagnostics import decompose_reward, run_reward_diagnostics


def make_checkpoints(directory):
    for index, name in enumerate(AGENT_NAMES):
        LinearDQNAgent(name, 7, seed=index).save(Path(directory) / f"{name}.npz")


class TestRewardDiagnostics(unittest.TestCase):

    def test_reward_decomposition_matches_current_formula(self):
        metrics = {
            "production_input": 520.0, "emissions": 1000.0,
            "operating_cost": 5000.0, "inventory": 600.0,
            "transit_days": 2.0, "net_waste": 10.0, "backlog": 100.0,
            "energy_used": 780.0, "service_level": 0.95, "otif": 0.95,
            "gross_scrap": 10.0, "recovered": 5.0,
        }
        parts = decompose_reward(metrics)
        self.assertAlmostEqual(
            parts["reconstructed_reward"],
            -parts["penalty_total"] + parts["bonus_total"],
        )
        self.assertEqual(parts["service_bonus"], 0.5)
        self.assertEqual(parts["otif_bonus"], 0.5)
        self.assertEqual(parts["recovery_bonus"], 0.25)

    def test_diagnostic_outputs_and_repeated_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "models"
            make_checkpoints(checkpoints)
            first = run_reward_diagnostics([1, 2], checkpoints, Path(directory) / "first")
            second = run_reward_diagnostics([1, 2], checkpoints, Path(directory) / "second")
            pd.testing.assert_frame_equal(first[0], second[0])
            pd.testing.assert_frame_equal(first[1], second[1])
            self.assertEqual(len(first[0]), 8)
            self.assertIn("penalty_sum", first[0].columns)
            self.assertIn("trained", set(first[0]["policy"]))

    def test_checkpoint_weights_are_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "models"
            make_checkpoints(checkpoints)
            before = {
                name: np.load(checkpoints / f"{name}.npz")["weights"].copy()
                for name in AGENT_NAMES
            }
            run_reward_diagnostics([3], checkpoints, Path(directory) / "results")
            for name in AGENT_NAMES:
                after = np.load(checkpoints / f"{name}.npz")["weights"]
                np.testing.assert_array_equal(before[name], after)


if __name__ == "__main__":
    unittest.main()