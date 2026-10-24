"""Smoke-train and train the five-agent paper-inspired MARL system."""

import argparse
import csv
from pathlib import Path

import numpy as np

from marl_agents import (
    AGENT_NAMES,
    LinearDQNAgent,
    build_observation,
)
from paper_environment import CircularSupplyChainDT


def create_agents(seed, episode_length):
    agents = {}
    for offset, name in enumerate(AGENT_NAMES):
        observation = build_observation(
            name,
            CircularSupplyChainDT(episode_length=episode_length, seed=seed).reset(),
            episode_length=episode_length,
        )
        agents[name] = LinearDQNAgent(
            name=name,
            observation_size=len(observation),
            seed=seed + offset,
        )
    return agents


def train(episodes=5, episode_length=10, seed=42, output_dir="marl_artifacts/smoke"):
    """Train a small reproducible run and save logs/checkpoints."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    agents = create_agents(seed, episode_length)
    rows = []

    for episode in range(episodes):
        env = CircularSupplyChainDT(
            episode_length=episode_length,
            seed=seed + episode,
        )
        state = env.reset(seed=seed + episode)
        previous_metrics = {}
        episode_reward = 0.0
        episode_losses = {name: [] for name in AGENT_NAMES}
        done = False

        while not done:
            observations = {
                name: build_observation(
                    name, state, previous_metrics, episode_length
                )
                for name in AGENT_NAMES
            }
            actions = {
                name: agents[name].select_action(observations[name])
                for name in AGENT_NAMES
            }
            next_state, metrics, done = env.step(actions)
            reward = float(metrics["reward"])
            if not np.isfinite(reward):
                raise FloatingPointError("Environment returned a non-finite reward")

            for name in AGENT_NAMES:
                next_observation = build_observation(
                    name, next_state, metrics, episode_length
                )
                agents[name].remember(
                    observations[name], actions[name], reward,
                    next_observation, done,
                )
                episode_losses[name].append(agents[name].train_step())

            episode_reward += reward
            state = next_state
            previous_metrics = metrics

        for agent in agents.values():
            agent.decay_exploration()
            agent.save(output_path / f"{agent.name}.npz")

        rows.append({
            "episode": episode + 1,
            "total_reward": episode_reward,
            "mean_loss": float(np.mean([
                loss for losses in episode_losses.values() for loss in losses
            ])),
            **{
                f"loss_{name}": float(np.mean(episode_losses[name]))
                for name in AGENT_NAMES
            },
            **{
                f"weight_norm_{name}": float(np.linalg.norm(agents[name].weights))
                for name in AGENT_NAMES
            },
            **{f"epsilon_{name}": agents[name].epsilon for name in AGENT_NAMES},
        })

    with (output_path / "training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="marl_artifacts/smoke")
    args = parser.parse_args()
    rows = train(**vars(args))
    print(f"Completed {len(rows)} episodes. Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
