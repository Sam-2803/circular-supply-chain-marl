
import numpy as np
import pandas as pd

from rl_environment import SupplyChainRLEnvironment


def discretize_state(state):
    day, inventory, backlog = state

    day_level = min(int(day * 5), 4)

    if inventory < 0.25:
        inventory_level = 0
    elif inventory < 0.60:
        inventory_level = 1
    else:
        inventory_level = 2

    if backlog < 0.10:
        backlog_level = 0
    elif backlog < 0.40:
        backlog_level = 1
    else:
        backlog_level = 2

    return (day_level, inventory_level, backlog_level)


# Load the trained Q-table.
q_df = pd.read_csv("q_table_results.csv")
q_table = {}

for _, row in q_df.iterrows():
    state = tuple(
        int(value.strip())
        for value in row["state"].strip("()").split(",")
    )

    q_table[state] = np.array([
        row["q_action_0"],
        row["q_action_1"],
        row["q_action_2"]
    ])


def evaluate_strategy(strategy, seed):
    env = SupplyChainRLEnvironment(
        seed=seed,
        episode_length=20
    )

    state = env.reset()
    records = []
    total_reward = 0.0

    for _ in range(env.episode_length):

        if strategy == "Low production":
            action = 0

        elif strategy == "Fixed production":
            action = 1

        else:
            discrete_state = discretize_state(state)

            if discrete_state in q_table:
                action = int(np.argmax(q_table[discrete_state]))
            else:
                action = 1

        next_state, reward, done, metrics = env.step(action)

        record = metrics.copy()
        record["action"] = action
        records.append(record)

        total_reward += reward
        state = next_state

        if done:
            break

    results = pd.DataFrame(records)

    return {
        "seed": seed,
        "strategy": strategy,
        "service_level": results["service_level"].mean(),
        "operating_cost": results["total_cost"].sum(),
        "co2_emissions": results["co2_emissions"].sum(),
        "ending_backlog": results["backlog"].iloc[-1],
        "ending_inventory": results["inventory"].iloc[-1],
        "total_reward": total_reward
    }


# Use the same 30 demand seeds for every strategy.
strategies = [
    "Low production",
    "Fixed production",
    "Q-learning"
]

all_results = []

for seed in range(100, 130):
    for strategy in strategies:
        all_results.append(
            evaluate_strategy(strategy, seed)
        )

details = pd.DataFrame(all_results)

# Save every individual evaluation.
details.to_csv("evaluation_all_runs.csv", index=False)

# Calculate average performance across 30 episodes.
summary = details.groupby("strategy").agg(
    average_service_level=("service_level", "mean"),
    average_operating_cost=("operating_cost", "mean"),
    average_co2_emissions=("co2_emissions", "mean"),
    average_ending_backlog=("ending_backlog", "mean"),
    average_ending_inventory=("ending_inventory", "mean"),
    average_reward=("total_reward", "mean"),
    reward_std=("total_reward", "std")
).reset_index()

summary.to_csv("evaluation_summary.csv", index=False)

print("\nAVERAGE RESULTS ACROSS 30 EVALUATION EPISODES")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "average_service_level": "{:.2%}".format,
            "average_operating_cost": "{:,.2f}".format,
            "average_co2_emissions": "{:,.2f}".format,
            "average_ending_backlog": "{:.2f}".format,
            "average_ending_inventory": "{:.2f}".format,
            "average_reward": "{:.3f}".format,
            "reward_std": "{:.3f}".format
        }
    )
)

print("\nSaved detailed results: evaluation_all_runs.csv")
print("Saved average results: evaluation_summary.csv")
