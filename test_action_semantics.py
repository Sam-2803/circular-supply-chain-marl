import unittest

from paper_environment import CircularSupplyChainDT


NEUTRAL_ACTIONS = {
    "planning": 1,
    "inventory": 1,
    "logistics": 1,
    "expediting": 0,
    "recycling": 1,
}


def run_one(actions, seed=17, initial_inventory=None):
    env = CircularSupplyChainDT(episode_length=3, seed=seed)
    env.reset(seed=seed)
    if initial_inventory is not None:
        env.inventory = float(initial_inventory)
    _, metrics, _ = env.step(actions)
    return metrics


class TestActionSemantics(unittest.TestCase):

    def test_default_action_space_and_neutral_mapping(self):
        metrics = run_one(NEUTRAL_ACTIONS)
        self.assertEqual(metrics["production_input"], 520.0)
        self.assertEqual(metrics["inventory_target"], 600.0)
        self.assertGreaterEqual(metrics["transit_days"], 1)

    def test_planning_levels_change_production_input(self):
        values = []
        for action in (0, 1, 2):
            actions = dict(NEUTRAL_ACTIONS, planning=action)
            values.append(run_one(actions)["production_input"])
        self.assertEqual(values, [468.0, 520.0, 572.0])

    def test_inventory_levels_change_reserve_target(self):
        values = []
        for action in (0, 1, 2):
            actions = dict(NEUTRAL_ACTIONS, inventory=action)
            values.append(run_one(actions)["inventory_target"])
        self.assertEqual(values, [500.0, 600.0, 700.0])

    def test_logistics_levels_are_current_dispatch_fractions(self):
        # The environment currently exposes fractions, not paper cadence.
        values = []
        for action in (0, 1, 2):
            actions = dict(NEUTRAL_ACTIONS, logistics=action)
            values.append(run_one(actions, initial_inventory=1000)["dispatched"])
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[2], values[0])

    def test_expediting_and_recycling_have_valid_boundaries(self):
        transit = [
            run_one(dict(NEUTRAL_ACTIONS, expediting=action))["transit_days"]
            for action in (0, 1, 2)
        ]
        recovered = [
            run_one(dict(NEUTRAL_ACTIONS, recycling=action))["recovered"]
            for action in (0, 1, 2)
        ]
        self.assertTrue(all(value >= 1 for value in transit))
        self.assertLessEqual(transit[2], transit[1])
        self.assertLessEqual(transit[1], transit[0])
        self.assertLessEqual(recovered[0], recovered[1])
        self.assertLessEqual(recovered[1], recovered[2])

    def test_invalid_action_boundaries_are_rejected(self):
        env = CircularSupplyChainDT(seed=1)
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(dict(NEUTRAL_ACTIONS, logistics=3))
        with self.assertRaises(ValueError):
            env.step(dict(NEUTRAL_ACTIONS, planning=-1))


if __name__ == "__main__":
    unittest.main()