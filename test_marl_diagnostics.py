import unittest

import numpy as np

from marl_agents import AGENT_NAMES
from marl_diagnostics import evaluate_policy, no_op_policy


class TestMARLDiagnostics(unittest.TestCase):

    def test_evaluation_does_not_require_exploration(self):
        rows, _ = evaluate_policy(
            "no_op", lambda _seed: no_op_policy, [101], episode_length=3
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isfinite(rows[0]["reward"]))

    def test_all_agent_names_are_present(self):
        self.assertEqual(len(AGENT_NAMES), 5)

    def test_seeded_exogenous_trajectories_are_policy_independent(self):
        from marl_diagnostics import evaluate_policy

        no_op_rows, no_op_trajectory = evaluate_policy(
            "no_op", lambda _seed: no_op_policy, [202], episode_length=5
        )
        random_rows, random_trajectory = evaluate_policy(
            "random", lambda seed: __import__("marl_diagnostics").random_policy(seed),
            [202], episode_length=5,
        )
        self.assertEqual(len(no_op_rows), len(random_rows))
        np.testing.assert_array_equal(no_op_trajectory[0], random_trajectory[0])


if __name__ == "__main__":
    unittest.main()