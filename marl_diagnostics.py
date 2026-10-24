"""Diagnostics and fair-seed evaluation for the paper-inspired MARL system."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marl_agents import AGENT_NAMES, LinearDQNAgent, build_observation
from marl_training import train
from paper_environment import CircularSupplyChainDT


OUTPUT_DIR = Path("marl_artifacts/training_100")


def load_agents(directory, episode_length):
    agents = {}
    state = CircularSupplyChainDT(episode_length=episode_length, seed=0).reset()
    for offset, name in enumerate(AGENT_NAMES):
        observation = build_observation(name, state, episode_length=episode_length)
        agent = LinearDQNAgent(name, len(observation), seed=offset)
        data = np.load(Path(directory) / f"{name}.npz")
        agent.weights = np.asarray(data["weights"], dtype=np.float64).copy()
        agent.epsilon = 0.0
        agents[name] = agent
    return agents


def learned_policy(agents, episode_length):
    previous_metrics = {}

    def policy(state, metrics=None):
        nonlocal previous_metrics
        metrics = metrics or previous_metrics
        actions = {
            name: agents[name].select_action(
                build_observation(name, state, metrics, episode_length),
                explore=False,
            )
            for name in AGENT_NAMES
        }
        previous_metrics = metrics
        return actions

    return policy


def no_op_policy(_state, _metrics=None):
    return {
        "planning": 1,
        "inventory": 1,
        "logistics": 1,
        "expediting": 0,
        "recycling": 1,
    }


def random_policy(seed):
    rng = np.random.default_rng(seed)

    def policy(_state, _metrics=None):
        return {name: int(rng.integers(0, 3)) for name in AGENT_NAMES}

    return policy


def heuristic_policy(_state, metrics=None):
    metrics = metrics or {}
    backlog = float(metrics.get("backlog", 0.0))
    service = float(metrics.get("service_level", 0.0))
    return {
        "planning": 2 if backlog > 900 else 1,
        "inventory": 0 if backlog > 900 else 1,
        "logistics": 2 if backlog > 900 else 1,
        "expediting": 2 if service < 0.90 else 0,
        "recycling": 2,
    }


def evaluate_policy(policy_name, policy_factory, seeds, episode_length=60):
    results = []
    trajectory_checks = []
    for seed in seeds:
        env = CircularSupplyChainDT(episode_length=episode_length, seed=seed)
        state = env.reset(seed=seed)
        policy = policy_factory(seed)
        metrics_rows = []
        previous_metrics = {}
        done = False
        while not done:
            actions = policy(state, previous_metrics)
            next_state, metrics, done = env.step(actions)
            metrics_rows.append(metrics)
            previous_metrics = metrics
            state = next_state

        frame = pd.DataFrame(metrics_rows)
        results.append({
            "policy": policy_name,
            "seed": seed,
            "cost": frame["operating_cost"].sum(),
            "emissions": frame["emissions"].sum(),
            "waste": frame["net_waste"].sum(),
            "backlog": frame.iloc[-1]["backlog"],
            "service_level": frame.iloc[-1]["service_level"],
            "otif": frame.iloc[-1]["otif"],
            "reward": frame["reward"].sum(),
        })
        trajectory_checks.append(
            frame[["demand", "yield_rate", "baseline_transit_days"]].to_numpy()
        )
    return results, trajectory_checks


def run_diagnostics(episodes=100, episode_length=60, seed=42):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = OUTPUT_DIR / "checkpoints"
    train(episodes, episode_length, seed, train_dir)
    log = pd.read_csv(train_dir / "training_log.csv")
    log.to_csv(OUTPUT_DIR / "training_log.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(log["episode"], log["total_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("MARL training reward")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "episode_reward.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    for name in AGENT_NAMES:
        plt.plot(log["episode"], log[f"loss_{name}"], label=name)
    plt.xlabel("Episode")
    plt.ylabel("Mean TD loss")
    plt.title("Per-agent training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "agent_loss.png", dpi=150)
    plt.close()

    initial_norms = {name: 0.0 for name in AGENT_NAMES}
    final_norms = {
        name: float(log[f"weight_norm_{name}"].iloc[-1])
        for name in AGENT_NAMES
    }
    with (OUTPUT_DIR / "parameter_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "initial_norm", "final_norm", "changed"])
        for name in AGENT_NAMES:
            writer.writerow([
                name, initial_norms[name], final_norms[name],
                bool(final_norms[name] > 0.0),
            ])

    agents = load_agents(train_dir, episode_length)
    seeds = range(1000, 1010)
    policies = {
        "trained": lambda _seed: learned_policy(agents, episode_length),
        "no_op": lambda _seed: no_op_policy,
        "random": random_policy,
        "heuristic": lambda _seed: heuristic_policy,
    }
    all_results = []
    trajectories = {}
    for name, factory in policies.items():
        rows, checks = evaluate_policy(name, factory, seeds, episode_length)
        all_results.extend(rows)
        trajectories[name] = checks

    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_DIR / "evaluation_runs.csv", index=False)
    summary = results.groupby("policy").agg(
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in ["cost", "emissions", "waste", "backlog", "service_level", "otif", "reward"]
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in ["cost", "emissions", "waste", "backlog", "service_level", "otif", "reward"]
        },
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)

    reference = trajectories["no_op"]
    comparability = []
    for policy, arrays in trajectories.items():
        demand_equal = all(np.array_equal(item[:, 0], ref[:, 0]) for item, ref in zip(arrays, reference))
        yield_equal = all(np.array_equal(item[:, 1], ref[:, 1]) for item, ref in zip(arrays, reference))
        transport_equal = all(np.array_equal(item[:, 2], ref[:, 2]) for item, ref in zip(arrays, reference))
        comparability.append({
            "policy": policy,
            "demand_equal_to_no_op": demand_equal,
            "yield_equal_to_no_op": yield_equal,
            "baseline_transport_equal_to_no_op": transport_equal,
        })
    pd.DataFrame(comparability).to_csv(OUTPUT_DIR / "scenario_comparability.csv", index=False)
    return summary


if __name__ == "__main__":
    summary = run_diagnostics()
    print(summary.to_string(index=False))