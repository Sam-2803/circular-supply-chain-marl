import unittest

from paper_config import (
    UnresolvedPaperSettingError,
    exploratory_config,
    paper_comparable_config,
)


class TestPaperConfig(unittest.TestCase):

    def test_paper_horizon_is_verified_as_twenty_days(self):
        config = paper_comparable_config()
        self.assertEqual(config.episode_horizon, 20)
        self.assertEqual(config.agent_names, (
            "planning", "inventory", "logistics", "expediting", "recycling"
        ))

    def test_verified_and_unresolved_settings_are_distinguishable(self):
        config = paper_comparable_config()
        self.assertTrue(config.shared_global_reward)
        self.assertTrue(config.linear_action_value)
        self.assertTrue(config.one_step_td_target)
        self.assertTrue(config.greedy_evaluation)
        self.assertTrue(config.learning_rate_range_is_approximate)
        self.assertIn("reward_weights", config.unresolved_settings)
        self.assertIn("observation_features", config.unresolved_settings)
        self.assertIn("epsilon_schedule", config.unresolved_settings)

    def test_unresolved_settings_reject_paper_faithful_validation(self):
        with self.assertRaises(UnresolvedPaperSettingError):
            paper_comparable_config().require_paper_faithful()

    def test_exploratory_configuration_preserves_provisional_settings(self):
        config = exploratory_config()
        self.assertEqual(config.mode, "exploratory")
        self.assertEqual(config.episode_horizon, 60)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.discount_factor, 0.95)
        self.assertEqual(config.replay_settings["batch_size"], 32)
        config.require_paper_faithful()


if __name__ == "__main__":
    unittest.main()