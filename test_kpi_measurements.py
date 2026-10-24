import unittest

import numpy as np

from kpi_measurements import measure_episode_kpis
from noop_normalization import noop_actions
from paper_environment import CircularSupplyChainDT


class TestKPIMeasurements(unittest.TestCase):

    def test_hand_calculated_completed_order_lead_time(self):
        env = CircularSupplyChainDT(episode_length=1, seed=1)
        env.reset(seed=1)
        env.orders = [{"created_day": 2, "completed_day": 5}]
        result = measure_episode_kpis(
            env, [{"operating_cost": 20.0, "delivered": 4.0}]
        )
        self.assertEqual(result["average_lead_time"], 3.0)
        self.assertEqual(result["cost_per_delivered_unit"], 5.0)
        self.assertEqual(result["completed_order_count"], 1)

    def test_empty_and_zero_denominator_cases_are_explicit(self):
        env = CircularSupplyChainDT(episode_length=1, seed=2)
        env.reset(seed=2)
        result = measure_episode_kpis(env, [])
        self.assertTrue(np.isnan(result["average_lead_time"]))
        self.assertTrue(np.isnan(result["cost_per_delivered_unit"]))
        self.assertTrue(np.isnan(result["wip"]))

    def test_incomplete_orders_are_censored_not_zero_lead_time(self):
        env = CircularSupplyChainDT(episode_length=1, seed=3)
        env.reset(seed=3)
        env.orders = [{"created_day": 1, "completed_day": None}]
        result = measure_episode_kpis(
            env, [{"operating_cost": 10.0, "delivered": 0.0}]
        )
        self.assertTrue(np.isnan(result["average_lead_time"]))
        self.assertEqual(result["censored_order_count"], 1)

    def test_fixed_seed_episode_measurement_is_deterministic(self):
        results = []
        for _ in range(2):
            env = CircularSupplyChainDT(episode_length=5, seed=8)
            env.reset(seed=8)
            rows = []
            done = False
            while not done:
                _, metrics, done = env.step(noop_actions())
                rows.append(metrics)
            results.append(measure_episode_kpis(env, rows))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()