
import numpy as np
import pandas as pd


class SupplyChainSimulator:

    def __init__(self, seed=42, episode_length=20):
        self.rng = np.random.default_rng(seed)
        self.episode_length = episode_length

        # Supply-chain settings
        self.production_capacity = 520
        self.initial_inventory = 500

        # Circular supply-chain settings
        self.return_rate = 0.08
        self.recovery_rate = 0.60
        self.recycling_capacity = 80

        # Cost assumptions
        self.production_cost_per_unit = 10
        self.holding_cost_per_unit = 0.5
        self.backlog_cost_per_unit = 5
        self.transport_cost_per_unit = 2

        # Emissions assumptions
        self.production_emissions_per_unit = 2
        self.transport_emissions_per_unit = 0.5

        # Default production decision
        self.production_multiplier = 1.0

        self.reset()

    def reset(self):
        self.day = 0
        self.inventory = self.initial_inventory
        self.backlog = 0
        self.history = []

        return self.get_state()

    def set_production_action(self, action):
        """
        Action 0: produce 90% of capacity.
        Action 1: produce 100% of capacity.
        Action 2: produce 110% of capacity.
        """

        multipliers = {
            0: 0.9,
            1: 1.0,
            2: 1.1,
        }

        if action not in multipliers:
            raise ValueError("Action must be 0, 1, or 2.")

        self.production_multiplier = multipliers[action]

    def get_state(self):
        return {
            "day": self.day,
            "inventory": self.inventory,
            "backlog": self.backlog,
        }

    def step(self):
        if self.day >= self.episode_length:
            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        # 1. Generate customer demand.
        demand = int(self.rng.integers(350, 651))

        # 2. Choose production based on the selected action.
        production = int(
            self.production_capacity
            * self.production_multiplier
        )

        # 3. Add production to available inventory.
        available = self.inventory + production

        # 4. Fulfil outstanding demand.
        previous_backlog = self.backlog
        total_required = previous_backlog + demand

        delivered = min(available, total_required)

        # Estimate how much of today's demand was fulfilled.
        served_today = min(
            demand,
            max(0, delivered - previous_backlog)
        )

        # 5. Update inventory and backlog.
        self.inventory = available - delivered
        self.backlog = total_required - delivered

        # 6. Calculate service level.
        service_level = served_today / demand

        # Simplified lead-time indicator.
        lead_time_proxy = 1 + self.backlog / max(demand, 1)

        # 7. Calculate returns, recovery and waste.
        returns = int(round(delivered * self.return_rate))

        recovered = min(
            int(round(returns * self.recovery_rate)),
            self.recycling_capacity
        )

        waste = returns - recovered

        # 8. Calculate operating costs.
        production_cost = (
            production * self.production_cost_per_unit
        )

        holding_cost = (
            self.inventory * self.holding_cost_per_unit
        )

        backlog_cost = (
            self.backlog * self.backlog_cost_per_unit
        )

        transport_cost = (
            delivered * self.transport_cost_per_unit
        )

        total_cost = (
            production_cost
            + holding_cost
            + backlog_cost
            + transport_cost
        )

        # 9. Calculate emissions and efficiency indicators.
        co2 = (
            production * self.production_emissions_per_unit
            + delivered * self.transport_emissions_per_unit
        )

        cost_per_unit = total_cost / max(delivered, 1)
        energy_per_unit = 1.5

        # 10. Store today's results.
        record = {
            "day": self.day + 1,
            "demand": demand,
            "production": production,
            "delivered": delivered,
            "inventory": self.inventory,
            "backlog": self.backlog,
            "service_level": service_level,
            "lead_time_proxy": lead_time_proxy,
            "returns": returns,
            "recovered_material": recovered,
            "material_waste": waste,
            "total_cost": total_cost,
            "co2_emissions": co2,
            "cost_per_unit": cost_per_unit,
            "energy_per_unit": energy_per_unit,
        }

        self.history.append(record)
        self.day += 1

        done = self.day >= self.episode_length

        return self.get_state(), record, done

    def run_episode(self):
        self.reset()

        while self.day < self.episode_length:
            self.step()

        return pd.DataFrame(self.history)


if __name__ == "__main__":

    # Compare all three production decisions.
    for action in [0, 1, 2]:

        sim = SupplyChainSimulator(
            seed=42,
            episode_length=20
        )

        sim.set_production_action(action)

        results = sim.run_episode()

        print("\n" + "=" * 40)
        print("Production action:", action)
        print(
            "Production per day:",
            int(
                sim.production_capacity
                * sim.production_multiplier
            )
        )

        print(
            f"Average service level: "
            f"{results['service_level'].mean():.2%}"
        )

        print(
            f"Total operating cost: "
            f"{results['total_cost'].sum():,.2f}"
        )

        print(
            f"Total CO2 emissions: "
            f"{results['co2_emissions'].sum():,.2f}"
        )

        print(
            "Ending inventory:",
            results["inventory"].iloc[-1]
        )

        print(
            "Ending backlog:",
            results["backlog"].iloc[-1]
        )

    # Save the baseline comparison results.
    print("\nComparison complete.")
