import unittest

from reward_audit import reward_audit_report, require_paper_reward_details


class TestRewardAudit(unittest.TestCase):

    def test_report_distinguishes_paper_and_exploratory_reward(self):
        report = reward_audit_report()
        self.assertEqual(report["paper"]["page"], 9)
        self.assertIn("normalized_KPI", report["paper"]["equation"])
        self.assertEqual(report["paper"]["weights"], "unresolved")
        self.assertEqual(report["exploratory"]["terms"]["co2"]["weight"], 1.1)
        self.assertEqual(report["exploratory"]["bonuses"]["otif_threshold"]["bonus"], 0.5)

    def test_paper_reward_validation_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Paper-faithful reward"):
            require_paper_reward_details()

    def test_report_does_not_change_existing_reward_configuration(self):
        before = reward_audit_report()["exploratory"]["terms"].copy()
        after = reward_audit_report()["exploratory"]["terms"].copy()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()