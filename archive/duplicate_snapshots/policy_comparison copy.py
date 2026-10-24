
import pandas as pd
from paper_environment import CircularSupplyChainDT


POLICIES = {
    "Neutral baseline": {
        "planning": 1,
        "inventory": 1,
        "logistics": 1,
        "expediting": 0,
        "recycling": 1,
    },
    "Higher production": {
        "planning": 2,
        "inventory": 1,
        "logistics": 1,
        "expediting": 0,
        "recycling": 1,
    },
    "Higher production + faster dispatch": {
        "planning": 2,
        "inventory": 1,
        "logistics": 2,
        "expediting": 0,
        "recycling": 1,
    },
    "Higher production + expediting": {
        "planning": 2,
        "inventory": 1,
        "logistics": 2,
        "expediting": 2,
        "recycling": 1,
    },
    "Higher production + more recycling": {
        "planning": 2,
        "inventory": 1,
        "logistics": 2,
        "expediting": 0,
        "recycling": 2,
    },
}


def evaluate_policy(policy_name, actions, seeds=range(10)):
    results = []

    for seed in seeds:
        env = CircularSupplyChainDT(
            episode_length=60,
            seed=seed,
        )
        env.reset(seed=seed)

        daily_metrics = []
        done = False

        while not done:
            _, metrics, done = env.step(actions)
            daily_metrics.append(metrics)

        df = pd.DataFrame(daily_metrics)

        results.append({
            "policy": policy_name,
            "seed": seed,
            "total_cost": df["operating_cost"].sum(),
            "total_emissions": df["emissions"].sum(),
            "total_waste": df["net_waste"].sum(),
            "final_backlog": df.iloc[-1]["backlog"],
            "final_service_level": df.iloc[-1]["service_level"],
            "final_otif": df.iloc[-1]["otif"],
            "average_reward": df["reward"].mean(),
        })

    return results


if __name__ == "__main__":
    all_results = []

    for name, actions in POLICIES.items():
        print(f"Evaluating: {name}")
        all_results.extend(
            evaluate_policy(name, actions)
        )

    df = pd.DataFrame(all_results)
    df.to_csv("policy_comparison_raw.csv", index=False)

    summary = (
        df.groupby("policy")
        .mean(numeric_only=True)
        .drop(columns=["seed"])
        .round(3)
    )

    print("\nPolicy comparison: averages across 10 seeds\n")
    print(summary.to_string())

    summary.to_csv("policy_comparison_summary.csv")

    print("\nSaved:")
    print("- policy_comparison_raw.csv")
    print("- policy_comparison_summary.csv")
