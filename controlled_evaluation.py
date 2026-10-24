"""Controlled, common-random-number evaluation for existing policies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kpi_measurements import measure_episode_kpis
from marl_agents import AGENT_NAMES
from marl_diagnostics import (
    heuristic_policy,
    learned_policy,
    load_agents,
    no_op_policy,
    random_policy,
)
from paper_config import paper_comparable_config
from paper_environment import CircularSupplyChainDT


METRICS = (
    "cost",
    "emissions",
    "waste",
    "backlog",
    "service_level",
    "otif",
    "average_inventory",
    "average_lead_time",
    "cost_per_delivered_unit",
    "completed_order_count",
    "censored_order_count",
    "wip",
    "reward_sum",
    "reward_mean",
)


def _policy_factories(checkpoint_dir, episode_length):
    if checkpoint_dir is None:
        raise ValueError("A trained checkpoint directory is required")
    agents = load_agents(checkpoint_dir, episode_length)
    return {
        "no_op": lambda _seed: no_op_policy,
        "random": random_policy,
        "heuristic": lambda _seed: heuristic_policy,
        "trained": lambda _seed: learned_policy(agents, episode_length),
    }


def evaluate_policy(policy_name, policy_factory, seed, episode_length):
    env = CircularSupplyChainDT(episode_length=episode_length, seed=seed)
    state = env.reset(seed=seed)
    policy = policy_factory(seed)
    rows = []
    previous_metrics = {}
    done = False
    while not done:
        actions = policy(state, previous_metrics)
        next_state, metrics, done = env.step(actions)
        rows.append(metrics)
        state = next_state
        previous_metrics = metrics

    frame = pd.DataFrame(rows)
    episode_kpis = measure_episode_kpis(env, rows)
    return {
        "policy": policy_name,
        "seed": seed,
        "episode_length": episode_length,
        "cost": float(frame["operating_cost"].sum()),
        "emissions": float(frame["emissions"].sum()),
        "waste": float(frame["net_waste"].sum()),
        "backlog": float(frame.iloc[-1]["backlog"]),
        "service_level": float(frame.iloc[-1]["service_level"]),
        "otif": float(frame.iloc[-1]["otif"]),
        "average_inventory": float(frame["inventory"].mean()),
        "average_lead_time": episode_kpis["average_lead_time"],
        "cost_per_delivered_unit": episode_kpis["cost_per_delivered_unit"],
        "completed_order_count": episode_kpis["completed_order_count"],
        "censored_order_count": episode_kpis["censored_order_count"],
        "wip": episode_kpis["wip"],
        "reward_sum": float(frame["reward"].sum()),
        "reward_mean": float(frame["reward"].mean()),
        "demand_path": frame["demand"].to_list(),
        "yield_path": frame["yield_rate"].to_list(),
        "transport_path": frame["baseline_transit_days"].to_list(),
    }


def run_controlled_evaluation(
    seeds,
    checkpoint_dir,
    output_dir="marl_artifacts/controlled_evaluation_stage4b",
    episode_length=None,
):
    """Evaluate all policies on identical fixed-seed exogenous paths."""
    config = paper_comparable_config()
    episode_length = config.episode_horizon if episode_length is None else episode_length
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a non-empty list of unique seeds")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    factories = _policy_factories(checkpoint_dir, episode_length)
    rows = []
    for policy_name, factory in factories.items():
        for seed in seeds:
            rows.append(evaluate_policy(policy_name, factory, seed, episode_length))

    detailed = pd.DataFrame(rows)
    path_columns = ["demand_path", "yield_path", "transport_path"]
    detailed.drop(columns=path_columns).to_csv(
        output_path / "evaluation_per_seed.csv", index=False
    )

    summary = detailed.groupby("policy").agg(
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in METRICS
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in METRICS
        },
    ).reset_index()
    summary.to_csv(output_path / "evaluation_summary.csv", index=False)

    reference = detailed[detailed["policy"] == "no_op"].set_index("seed")
    alignment = []
    for row in rows:
        reference_row = reference.loc[row["seed"]]
        alignment.append({
            "policy": row["policy"],
            "seed": row["seed"],
            "demand_equal_to_no_op": row["demand_path"] == reference_row["demand_path"],
            "yield_equal_to_no_op": row["yield_path"] == reference_row["yield_path"],
            "transport_equal_to_no_op": row["transport_path"] == reference_row["transport_path"],
        })
    alignment_frame = pd.DataFrame(alignment)
    alignment_frame.to_csv(output_path / "scenario_alignment.csv", index=False)

    metadata = {
        "episode_length": episode_length,
        "seeds": list(seeds),
        "policies": list(factories),
        "checkpoint_dir": str(checkpoint_dir),
        "evaluation_mode": "inference-only; trained agents use explore=False",
        "common_random_numbers": {
            "demand": "pre-seeded environment stream",
            "yield": "pre-seeded environment stream",
            "transport": "day-indexed pre-seeded latent stream",
        },
        "metric_notes": {
            "reward_sum": "sum of daily environment rewards",
            "reward_mean": "mean of daily environment rewards",
            "cost_per_delivered_unit": "proxy; not paper-defined cost per unit",
            "wip": "unavailable; production-stage WIP is not tracked",
            "average_lead_time": "completed orders only; censored orders counted separately",
        },
        "limitations": [
            "Exploratory reward and action semantics remain unchanged.",
            "This is not an exact paper reproduction.",
            "Existing checkpoints are evaluated; no retraining occurs.",
        ],
    }
    (output_path / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return detailed.drop(columns=path_columns), summary, alignment_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="marl_artifacts/controlled_evaluation_stage4b")
    parser.add_argument("--episode-length", type=int, default=20)
    args = parser.parse_args()
    _, summary, alignment = run_controlled_evaluation(
        args.seeds, args.checkpoint_dir, args.output_dir, args.episode_length
    )
    print(summary.to_string(index=False))
    print("\nScenario alignment:")
    print(alignment.to_string(index=False))


if __name__ == "__main__":
    main()