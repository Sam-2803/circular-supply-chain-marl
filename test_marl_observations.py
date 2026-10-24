import unittest

import numpy as np

from marl_observations import (
    NormalizerConfig,
    action_feature_map,
    build_observation,
)
from paper_environment import CircularSupplyChainDT


class TestMARLObservations(unittest.TestCase):

    def setUp(self):
        env = CircularSupplyChainDT(episode_length=20, seed=4)
        self.state = env.reset(seed=4)
        _, self.metrics, _ = env.step({
            "planning": 1, "inventory": 1, "logistics": 1,
            "expediting": 0, "recycling": 1,
        })
        self.normalizers = NormalizerConfig({
            "service_level": 1.0,
            "otif": 1.0,
            "operating_cost": 15000.0,
            "co2_emissions": 5000.0,
            "energy_per_unit": 2.0,
            "material_waste": 20.0,
            "average_inventory": 600.0,
            "backlog": 1500.0,
        })

    def test_feature_order_and_action_encoding_are_stable(self):
        first = build_observation(
            "planning", self.state, self.metrics, self.normalizers, "full_data"
        )
        second = build_observation(
            "planning", self.state, self.metrics, self.normalizers, "full_data"
        )
        self.assertEqual(first.feature_names, second.feature_names)
        np.testing.assert_array_equal(first.values, second.values)
        encoded = action_feature_map(first, 2)
        self.assertEqual(tuple(encoded.values[-3:]), (0.0, 0.0, 1.0))

    def test_full_data_and_silo_data_have_finite_distinct_features(self):
        full = build_observation(
            "planning", self.state, self.metrics, self.normalizers, "full_data"
        )
        silo = build_observation(
            "planning", self.state, self.metrics, self.normalizers, "silo_data"
        )
        self.assertTrue(np.isfinite(full.values).all())
        self.assertTrue(np.isfinite(silo.values).all())
        self.assertNotEqual(full.feature_names, silo.feature_names)

    def test_missing_metrics_are_reported_not_zero_filled(self):
        result = build_observation(
            "logistics", self.state, {}, self.normalizers, "full_data"
        )
        self.assertIn("operating_cost", result.missing_metrics)
        self.assertIn("average_lead_time", result.missing_metrics)
        self.assertTrue(np.isfinite(result.values).all())

    def test_missing_normalizer_is_reported(self):
        result = build_observation(
            "recycling", self.state, self.metrics,
            NormalizerConfig({"service_level": 1.0}), "full_data"
        )
        self.assertIn("material_waste", result.missing_metrics)


if __name__ == "__main__":
    unittest.main()