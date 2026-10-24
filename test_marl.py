import unittest

import numpy as np

from marl_agents import AGENT_NAMES, LinearDQNAgent, build_observation
from marl_training import train
from paper_environment import CircularSupplyChainDT


class TestMARLSmoke(unittest.TestCase):

    def test_each_agent_has_valid_action_and_finite_observation(self):
        env = CircularSupplyChainDT(episode_length=3, seed=7)
        state = env.reset()
        for index, name in enumerate(AGENT_NAMES):
            observation = build_observation(name, state, episode_length=3)
            agent = LinearDQNAgent(name, len(observation), seed=index)
            action = agent.select_action(observation)
            self.assertIn(action, (0, 1, 2))
            self.assertTrue(np.isfinite(observation).all())

    def test_small_training_run_produces_artifacts(self):
        rows = train(
            episodes=1,
            episode_length=3,
            seed=11,
            output_dir="marl_artifacts/test",
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isfinite(rows[0]["total_reward"]))


if __name__ == "__main__":
    unittest.main()
