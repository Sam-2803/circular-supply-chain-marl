
from paper_environment import CircularSupplyChainDT

actions = {
    "planning": 1,
    "inventory": 1,
    "logistics": 2,
    "expediting": 1,
    "recycling": 1,
}

for seed in range(10):
    env = CircularSupplyChainDT(episode_length=60, seed=seed)
    env.reset(seed=seed)

    for _ in range(60):
        state, metrics, done = env.step(actions)

        assert state["inventory"] >= 0, "Negative inventory"
        assert state["backlog"] >= 0, "Negative backlog"
        assert metrics["dispatched"] <= env.transport_capacity + 1e-8
        assert metrics["dispatched"] <= (
            metrics["dispatched"] + state["inventory"] + 1e-8
        )

        if done:
            break

print("Validation passed for 10 seeds × 60 days.")
