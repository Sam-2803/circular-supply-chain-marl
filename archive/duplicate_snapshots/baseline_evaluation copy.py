
import pandas as pd
from paper_environment import CircularSupplyChainDT


def run_baseline(seed, episode_length=60):
    env = CircularSupplyChainDT(
        episode_length=episode_length,
        seed=seed,
    )
    env.reset(seed=seed)

    # No-Op baseline: keep all agents at their neutral action.
    actions = {
        "planning": 1,
        "inventory": 1,
        "logistics": 1,
        "expediting": 0,
        "recycling": 1,
    }

    daily_metrics = []

    done = False
    while not done:
        _, metrics, done = env.step(actions)
        daily_metrics.append(metrics)

    df = pd.DataFrame(daily_metrics)

    return {
        "seed": seed,
        "total_cost": df["operating_cost"].sum(),
        "total_emissions": df["emissions"].sum(),
        "total_waste": df["net_waste"].sum(),
        "final_inventory": df.iloc[-1]["inventory"],
        "final_backlog": df.iloc[-1]["backlog"],
        "final_service_level": df.iloc[-1]["service_level"],
        "final_otif": df.iloc[-1]["otif"],
        "average_reward": df["reward"].mean(),
    }


if __name__ == "__main__":
    results = []

    for seed in range(10):
        results.append(run_baseline(seed))

    df = pd.DataFrame(results)
    df.to_csv("no_op_baseline.csv", index=False)

    print("\nNo-Op baseline results across 10 seeds:\n")
    print(df.round(3).to_string(index=False))

    print("\nMean metrics:\n")
    print(df.drop(columns=["seed"]).mean().round(3))

    print("\nSaved results to no_op_baseline.csv")
