import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from noop_period_diagnostic import build_normalizer_summary, run_diagnostic


class TestNoOpPeriodDiagnostic(unittest.TestCase):

    def test_row_count_and_seed_period_uniqueness(self):
        with tempfile.TemporaryDirectory() as directory:
            panel, orders, summary = run_diagnostic([100, 101], directory, 20)
        self.assertEqual(len(panel), 40)
        self.assertEqual(panel[["seed", "period"]].drop_duplicates().shape[0], 40)
        self.assertEqual(set(panel["seed"]), {100, 101})
        self.assertGreater(len(orders), 0)
        self.assertIn("service_level", set(summary["metric"]))

    def test_repeated_runs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            panel_a, orders_a, summary_a = run_diagnostic([100, 101], first, 20)
            panel_b, orders_b, summary_b = run_diagnostic([100, 101], second, 20)
        pd.testing.assert_frame_equal(panel_a, panel_b)
        pd.testing.assert_frame_equal(orders_a, orders_b)
        pd.testing.assert_frame_equal(summary_a, summary_b)

    def test_missing_values_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            panel, _, summary = run_diagnostic([100], directory, 20)
        self.assertTrue(panel["wip"].isna().all())
        self.assertTrue(panel["cost_per_unit"].isna().all())
        self.assertTrue(panel["schedule_adherence"].isna().all())
        self.assertEqual(
            summary.loc[summary["metric"] == "wip", "status"].iloc[0],
            "unavailable_or_invalid",
        )

    def test_zero_denominator_ratio_is_nan(self):
        panel = pd.DataFrame({
            "operating_cost": [10.0],
            "co2_emissions": [1.0],
            "inventory": [1.0],
            "material_waste": [1.0],
            "backlog": [1.0],
            "energy_used": [0.0],
            "production_input": [0.0],
            "energy_per_unit": [np.nan],
            "on_time_delivered_units": [0.0],
            "demand_numerator": [0.0],
            "service_level_period": [np.nan],
            "otif_orders": [0.0],
            "new_orders": [0.0],
            "otif_period": [np.nan],
        })
        orders = pd.DataFrame({"lead_time_days": [np.nan]})
        summary = build_normalizer_summary(panel, orders)
        for metric in ("energy_per_unit", "service_level", "otif"):
            row = summary[summary["metric"] == metric].iloc[0]
            self.assertTrue(pd.isna(row["candidate_normalizer"]))
            self.assertEqual(row["status"], "unavailable_or_invalid")

    def test_ratio_aggregation_uses_totals(self):
        panel = pd.DataFrame({
            "operating_cost": [1.0, 1.0], "co2_emissions": [1.0, 1.0],
            "inventory": [1.0, 1.0], "material_waste": [1.0, 1.0],
            "backlog": [1.0, 1.0], "energy_used": [2.0, 8.0],
            "production_input": [1.0, 4.0], "energy_per_unit": [2.0, 2.0],
            "on_time_delivered_units": [1.0, 3.0], "demand_numerator": [2.0, 6.0],
            "service_level_period": [0.5, 0.5], "otif_orders": [1.0, 1.0],
            "new_orders": [2.0, 8.0], "otif_period": [0.5, 0.125],
        })
        orders = pd.DataFrame({"lead_time_days": [1.0]})
        summary = build_normalizer_summary(panel, orders)
        values = summary.set_index("metric")["candidate_normalizer"]
        self.assertEqual(values["energy_per_unit"], 2.0)
        self.assertEqual(values["service_level"], 0.5)
        self.assertEqual(values["otif"], 0.2)


if __name__ == "__main__":
    unittest.main()