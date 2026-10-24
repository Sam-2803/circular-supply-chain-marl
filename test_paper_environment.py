
from paper_environment import CircularSupplyChainDT

env = CircularSupplyChainDT(seed=42)
state = env.reset()

actions = {
    "planning": 1,
    "inventory": 1,
    "logistics": 1,
    "expediting": 0,
    "recycling": 1
}

for day in range(20):
    state, metrics, done = env.step(actions)

    print(
        f"Day {day + 1:02d} | "
        f"Demand: {metrics['demand']:.0f} | "
        f"Production: {metrics['usable_output']:.1f} | "
        f"Delivered: {metrics['delivered']:.1f} | "
        f"Inventory: {metrics['inventory']:.1f} | "
        f"Backlog: {metrics['backlog']:.1f} | "
        f"Cost: {metrics['operating_cost']:.2f} | "
        f"CO2: {metrics['co2_emissions']:.2f}"
    )

    assert metrics["inventory"] >= 0, "Negative inventory detected"
    assert metrics["backlog"] >= 0, "Negative backlog detected"

    if done:
        break

print("\n20-day episode completed.")
print("Final state:", state)
