
import numpy as np
import pandas as pd

from rl_environment import SupplyChainRLEnvironment


# Convert the continuous environment state into discrete categories.
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


# Reproducible random choices.
rng = np.random.default_rng(42)

# Create the supply chain environment.
env = SupplyChainRLEnvironment(seed=42, episode_length=20)

# Q-table: one row per state, three possible actions per state.
q_table = {}

# Learning settings.
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
minimum_epsilon = 0.05

number_of_episodes = 1000
training_history = []

for episode in range(number_of_episodes):
    state = discretize_state(env.reset())
    total_reward = 0.0

    for day in range(env.episode_length):

        # Create a Q-table entry the first time we encounter a state.
        if state not in q_table:
            q_table[state] = np.zeros(3)

        # Epsilon-greedy action selection:
        # explore randomly or choose the action with the highest Q-value.
        if rng.random() < epsilon:
            action = int(rng.integers(0, 3))
        else:
            action = int(np.argmax(q_table[state]))

        next_state_raw, reward, done, metrics = env.step(action)
        next_state = discretize_state(next_state_raw)

        if next_state not in q_table:
            q_table[next_state] = np.zeros(3)

        # Q-learning update.
        old_value = q_table[state][action]

        if done:
            best_future_value = 0.0
        else:
            best_future_value = np.max(q_table[next_state])

        new_value = old_value + learning_rate * (
            reward
            + discount_factor * best_future_value
            - old_value
        )

        q_table[state][action] = new_value

        total_reward += reward
        state = next_state

        if done:
            break

    epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

    training_history.append({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon
    })

    if (episode + 1) % 100 == 0:
        recent_rewards = [
            row["total_reward"]
            for row in training_history[-100:]
        ]
        print(
            f"Episode {episode + 1:4d} | "
            f"Average reward (last 100): "
            f"{np.mean(recent_rewards):.3f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# Save training results.
training_df = pd.DataFrame(training_history)
training_df.to_csv("training_results.csv", index=False)

# Save the learned Q-table in a readable format.
q_rows = []

for state, values in q_table.items():
    q_rows.append({
        "state": str(state),
        "q_action_0": values[0],
        "q_action_1": values[1],
        "q_action_2": values[2],
        "best_action": int(np.argmax(values))
    })

pd.DataFrame(q_rows).to_csv("q_table_results.csv", index=False)

print("\nTraining complete.")
print("Episodes trained:", number_of_episodes)
print("States learned:", len(q_table))
print("Saved: training_results.csv")
print("Saved: q_table_results.csv")
