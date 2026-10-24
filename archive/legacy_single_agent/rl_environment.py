
import numpy as np

from supply_chain import SupplyChainSimulator


class SupplyChainRLEnvironment:

    def __init__(self, seed=42, episode_length=20):
        self.simulator = SupplyChainSimulator(
            seed=seed,
            episode_length=episode_length
        )

        self.episode_length = episode_length

        # Three possible production decisions:
        # 0 = 468 units, 1 = 520 units, 2 = 572 units.
        self.actions = [0, 1, 2]

    def reset(self):
        """Start a new episode and return the initial state."""
        self.simulator.reset()
        self.simulator.production_multiplier = 1.0

        return self.get_state()

    def get_state(self):
        """Represent the current situation using numeric values."""
        sim = self.simulator

        return np.array([
            sim.day / self.episode_length,
            sim.inventory / 1500,
            sim.backlog / 1500,
        ], dtype=np.float32)

    def step(self, action):
        """Apply an action and calculate its reward."""

        if action not in self.actions:
            raise ValueError("Action must be 0, 1, or 2.")

        # Apply the production decision.
        self.simulator.set_production_action(action)

        # Advance the supply chain by one day.
        next_state, metrics, done = self.simulator.step()

        # Reward components.
        service_reward = 10 * metrics["service_level"]

        cost_penalty = metrics["total_cost"] / 1000
        emissions_penalty = metrics["co2_emissions"] / 1000
        waste_penalty = metrics["material_waste"] / 100

        # Higher reward is better.
        reward = (
            service_reward
            - cost_penalty
            - emissions_penalty
            - waste_penalty
        )

        # Return a standard RL transition.
        return (
            self.get_state(),
            float(reward),
            done,
            metrics
        )


if __name__ == "__main__":

    env = SupplyChainRLEnvironment(seed=42)
    state = env.reset()

    total_reward = 0

    print("Initial state:", state)

    for day in range(env.episode_length):

        # Temporary test policy: always choose normal production.
        action = 1

        next_state, reward, done, metrics = env.step(action)

        total_reward += reward

        print(
            f"Day {day + 1:02d} | "
            f"Action: {action} | "
            f"Service: {metrics['service_level']:.2%} | "
            f"Reward: {reward:.3f}"
        )

        state = next_state

        if done:
            break

    print(f"\nTotal episode reward: {total_reward:.3f}")
