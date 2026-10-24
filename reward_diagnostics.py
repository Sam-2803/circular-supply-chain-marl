"""Read-only reward and action diagnostics for existing policy artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from controlled_evaluation import _policy_factories
from kpi_measurements import measure_episode_kpis
from paper_environment import CircularSupplyChainDT


REWARD_TERMS = {
    "co2": ("emissions", 5000.0, 1.1),
    "cost": ("operating_cost", 15000.0, 1.2),
    "inventory": ("inventory", 600.0, 0.5),
    "lead_time": ("transit_days", 3.0, 0.7),
    "waste": ("net_waste", 20.0, 1.2),
    "backlog": ("backlog", 1500.0, 0.6),
}


def decompose_reward(metrics):
    """Reconstruct the current reward without changing environment state."""
    production = max(float(metrics["production_input"]), 1.0)
    raw_values = {
        name: float(metrics[field]) / normalizer
        for name, (field, normalizer, _weight) in REWARD_TERMS.items()
    }
    raw_values["cost_per_unit"] = float(metrics["operating_cost"]) / production / 30.0
    raw_values["energy_per_unit"] = float(metrics["energy_used"]) / production / 2.0
    weights = {
        name: weight
        for name, (_field, _normalizer, weight) in REWARD_TERMS.items()
    }
    weights.update({"cost_per_unit": 1.0, "energy_per_unit": 0.6})
    clipped = {name: min(value, 2.0) for name, value in raw_values.items()}
    penalties = {name: weights[name] * clipped[name] for name in clipped}
    service_bonus = 0.5 if metrics["service_level"] >= 0.95 else 0.0
    otif_bonus = 0.5 if metrics["otif"] >= 0.95 else 0.0
    recovery_ratio = (
        float(metrics["recovered"]) / float(metrics["gross_scrap"])
        if metrics["gross_scrap"] > 0 else 0.0
    )
    recovery_bonus = 0.25 if recovery_ratio >= 0.4 else 0.0
    result = {
        **{f"raw_{name}": value for name, value in raw_values.items()},
        **{f"clipped_{name}": value for name, value in clipped.items()},
        **{f"penalty_{name}": value for name, value in penalties.items()},
        "penalty_total": float(sum(penalties.values())),
        "service_bonus": service_bonus,
        "otif_bonus": otif_bonus,
        "recovery_bonus": recovery_bonus,
        "bonus_total": service_bonus + otif_bonus + recovery_bonus,
        "recovery_ratio": recovery_ratio,
        "reconstructed_reward": -sum(penalties.values()) + service_bonus + otif_bonus + recovery_bonus,
    }
    for name, value in raw_values.items():
        result[f"clipped_flag_{name}"] = bool(value > 2.0)
    return result


def run_reward_diagnostics(
    seeds,
    checkpoint_dir,
    output_dir="marl_artifacts/reward_diagnostics_stage4c",
    episode_length=20,
):
    """Replay policies and save reward/action evidence without training."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a non-empty list of unique seeds")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    factories = _policy_factories(checkpoint_dir, episode_length)
    episode_rows = []
    step_rows = []

    for policy_name, factory in factories.items():
        for seed in seeds:
            env = CircularSupplyChainDT(episode_length=episode_length, seed=seed)
            state = env.reset(seed=seed)
            policy = factory(seed)
            previous_metrics = {}
            daily_metrics = []
            done = False
            while not done:
                actions = policy(state, previous_metrics)
                next_state, metrics, done = env.step(actions)
                parts = decompose_reward(metrics)
                step_rows.append({
                    "policy": policy_name,
                    "seed": seed,
                    "day": metrics["day"],
                    **{f"action_{name}": actions[name] for name in actions},
                    "reward": metrics["reward"],
                    **parts,
                    "backlog": metrics["backlog"],
                    "service_level": metrics["service_level"],
                    "otif": metrics["otif"],
                    "dispatched": metrics["dispatched"],
                    "inventory_target": metrics["inventory_target"],
                    "inventory": metrics["inventory"],
                    "production_input": metrics["production_input"],
                    "net_waste": metrics["net_waste"],
                    "emissions": metrics["emissions"],
                })
                daily_metrics.append(metrics)
                state = next_state
                previous_metrics = metrics

            frame = pd.DataFrame(daily_metrics)
            parts = pd.DataFrame([decompose_reward(row) for row in daily_metrics])
            kpis = measure_episode_kpis(env, daily_metrics)
            episode_row = {
                "policy": policy_name,
                "seed": seed,
                "episode_length": episode_length,
                "reward_sum": float(frame["reward"].sum()),
                "reward_mean": float(frame["reward"].mean()),
                "reconstructed_reward_sum": float(parts["reconstructed_reward"].sum()),
                "penalty_sum": float(parts["penalty_total"].sum()),
                "bonus_sum": float(parts["bonus_total"].sum()),
                "service_bonus_count": int((parts["service_bonus"] > 0).sum()),
                "otif_bonus_count": int((parts["otif_bonus"] > 0).sum()),
                "recovery_bonus_count": int((parts["recovery_bonus"] > 0).sum()),
                "cost": float(frame["operating_cost"].sum()),
                "emissions": float(frame["emissions"].sum()),
                "waste": float(frame["net_waste"].sum()),
                "backlog_final": float(frame.iloc[-1]["backlog"]),
                "service_level_final": float(frame.iloc[-1]["service_level"]),
                "otif_final": float(frame.iloc[-1]["otif"]),
                "average_inventory": float(frame["inventory"].mean()),
                **kpis,
            }
            recent_steps = step_rows[-episode_length:]
            for agent in actions:
                for action in (0, 1, 2):
                    episode_row[f"{agent}_action_{action}_count"] = sum(
                        row[f"action_{agent}"] == action for row in recent_steps
                    )
            episode_rows.append(episode_row)

    episodes = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    episodes.to_csv(output_path / "episode_diagnostics.csv", index=False)
    steps.to_csv(output_path / "step_diagnostics.csv", index=False)
    summary_metrics = [
        "reward_sum", "penalty_sum", "bonus_sum", "cost", "emissions",
        "waste", "backlog_final", "service_level_final", "otif_final",
    ]
    summary = episodes.groupby("policy").agg(
        **{f"{metric}_mean": (metric, "mean") for metric in summary_metrics},
        **{f"{metric}_std": (metric, "std") for metric in summary_metrics},
    ).reset_index()
    summary.to_csv(output_path / "reward_operational_summary.csv", index=False)
    action_names = [f"action_{name}" for name in actions]
    action_summary = steps.groupby(["policy"] + action_names).agg(
        observations=("day", "size"),
        mean_reward=("reward", "mean"),
        mean_backlog=("backlog", "mean"),
        mean_service_level=("service_level", "mean"),
        mean_otif=("otif", "mean"),
        mean_dispatched=("dispatched", "mean"),
        mean_inventory_target=("inventory_target", "mean"),
        mean_waste=("net_waste", "mean"),
    ).reset_index()
    action_summary.to_csv(output_path / "action_outcome_summary.csv", index=False)
    metadata = {
        "seeds": list(seeds),
        "episode_length": episode_length,
        "policies": list(factories),
        "checkpoint_dir": str(checkpoint_dir),
        "mode": "read-only replay; no training or weight updates",
        "limitations": [
            "Action/outcome summaries are associations, not causal effects.",
            "Reward weights and formula remain provisional.",
            "WIP remains unavailable.",
        ],
    }
    (output_path / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return episodes, steps, summary, action_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="marl_artifacts/reward_diagnostics_stage4c")
    parser.add_argument("--episode-length", type=int, default=20)
    args = parser.parse_args()
    _, _, summary, _ = run_reward_diagnostics(
        args.seeds, args.checkpoint_dir, args.output_dir, args.episode_length
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()