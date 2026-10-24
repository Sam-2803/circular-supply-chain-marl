
import unittest
from supply_chain import SupplyChainSimulator


class TestSupplyChain(unittest.TestCase):

    def test_episode_length(self):
        sim = SupplyChainSimulator(seed=42, episode_length=20)
        results = sim.run_episode()
        self.assertEqual(len(results), 20)

    def test_demand_range(self):
        sim = SupplyChainSimulator(seed=42, episode_length=20)
        results = sim.run_episode()
        self.assertTrue(results["demand"].between(350, 650).all())

    def test_no_negative_inventory_or_backlog(self):
        sim = SupplyChainSimulator(seed=42, episode_length=20)
        results = sim.run_episode()
        self.assertTrue((results["inventory"] >= 0).all())
        self.assertTrue((results["backlog"] >= 0).all())

    def test_service_level_range(self):
        sim = SupplyChainSimulator(seed=42, episode_length=20)
        results = sim.run_episode()
        self.assertTrue(results["service_level"].between(0, 1).all())

    def test_recycling_balance(self):
        sim = SupplyChainSimulator(seed=42, episode_length=20)
        results = sim.run_episode()
        self.assertTrue(
            (results["recovered_material"] <= results["returns"]).all()
        )
        self.assertTrue((results["material_waste"] >= 0).all())
        self.assertTrue(
            (
                results["recovered_material"]
                + results["material_waste"]
                == results["returns"]
            ).all()
        )

    def test_same_seed_reproduces_results(self):
        sim1 = SupplyChainSimulator(seed=123, episode_length=20)
        sim2 = SupplyChainSimulator(seed=123, episode_length=20)
        self.assertTrue(sim1.run_episode().equals(sim2.run_episode()))

    def test_episode_stops(self):
        sim = SupplyChainSimulator(seed=42, episode_length=2)
        sim.step()
        sim.step()

        with self.assertRaises(RuntimeError):
            sim.step()


if __name__ == "__main__":
    unittest.main()
