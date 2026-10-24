import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from noop_normalization import collect_noop_references, select_candidate_normalizers


class TestNoOpNormalization(unittest.TestCase):

    def test_same_seeds_are_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_values, first_summary = collect_noop_references([3, 7], first)
            second_values, second_summary = collect_noop_references([3, 7], second)
        pd.testing.assert_frame_equal(first_values, second_values)
        pd.testing.assert_frame_equal(first_summary, second_summary)

    def test_available_normalizers_are_positive_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            _, summary = collect_noop_references([1, 2, 3], directory)
        available = summary[summary["status"] == "available"]
        normalizer_rows = available[
            ~available["metric"].isin({
                "cost_per_delivered_unit",
                "completed_order_count",
                "censored_order_count",
            })
        ]
        self.assertTrue((normalizer_rows["mean"] > 0).all())
        self.assertTrue(np.isfinite(normalizer_rows["mean"]).all())
        proposal = select_candidate_normalizers(summary)
        self.assertEqual(proposal["status"], "candidate/provisional")
        self.assertTrue(proposal["fixed_within_evaluation_run"])

    def test_unavailable_metrics_are_reported_without_zero_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            per_seed, summary = collect_noop_references([5], directory)
        unavailable = summary[summary["status"] == "unavailable"]
        self.assertEqual(set(unavailable["metric"]), {"cost_per_unit", "wip"})
        for metric in unavailable["metric"]:
            self.assertTrue(pd.isna(per_seed.iloc[0][metric]))

    def test_diagnostic_does_not_change_existing_artifact_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            collect_noop_references([9], Path(directory) / "diagnostic")
            self.assertFalse((Path(directory) / "no_op_baseline.csv").exists())

    def test_episode_kpis_are_present_in_per_seed_and_summary_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            per_seed, summary = collect_noop_references([5], directory)
        for column in (
            "average_lead_time",
            "cost_per_delivered_unit",
            "completed_order_count",
            "censored_order_count",
        ):
            self.assertIn(column, per_seed.columns)
            self.assertIn(column, set(summary["metric"]))
        self.assertIn("wip", set(summary["metric"]))
        proxy_row = summary[
            summary["metric"] == "cost_per_delivered_unit"
        ].iloc[0]
        self.assertIn("proxy", proxy_row["normalizer_status"])

    def test_censored_counts_and_zero_delivery_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            per_seed, summary = collect_noop_references([5], directory)
        self.assertGreaterEqual(per_seed.iloc[0]["censored_order_count"], 0)
        self.assertTrue(np.isfinite(per_seed.iloc[0]["completed_order_count"]))
        wip_row = summary[summary["metric"] == "wip"].iloc[0]
        self.assertEqual(wip_row["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()