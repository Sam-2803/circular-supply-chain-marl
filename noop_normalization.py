"""Measure provisional No-Op KPI references without changing reward logic."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import paper_comparable_config
from paper_environment import CircularSupplyChainDT
from kpi_measurements import measure_episode_kpis


KPI_DEFINITIONS = {
    "average_lead_time": (
        "mean completed-order completion_day minus created_day, in days; "
        "incomplete orders are censored"
    ),
    "service_level": "final cumulative on-time units divided by cumulative demand",
    "otif": "final completed orders delivered fully by due date divided by total orders",
    "operating_cost": "sum of daily operating_cost values",
    "cost_per_unit": (
        "unavailable: the paper-defined unit denominator is not available"
    ),
    "cost_per_delivered_unit": (
        "proxy: total operating cost divided by total delivered units"
    ),
    "co2_emissions": "sum of daily emissions values",
    "energy_per_unit": "sum energy_used divided by sum production_input",
    "material_waste": "sum of daily net_waste values",
    "average_inventory": "mean of daily inventory values",
    "wip": "unavailable: WIP is not tracked by the environment",
    "backlog": "mean of daily backlog values",
    "completed_order_count": "number of fully completed orders in the episode",
    "censored_order_count": "number of orders incomplete at the episode boundary",
}

UNAVAILABLE_KPIS = {
    "cost_per_unit",
    "wip",
}

NON_NORMALIZER_METRICS = {
    "cost_per_delivered_unit",
    "completed_order_count",
    "censored_order_count",
}


def noop_actions():
    return {
        "planning": 1,
        "inventory": 1,
        "logistics": 1,
        "expediting": 0,
        "recycling": 1,
    }


def run_noop_episode(seed, episode_length=None):
    """Run one fixed-seed No-Op episode and return diagnostic KPI values."""
    if episode_length is None:
        episode_length = paper_comparable_config().episode_horizon
    env = CircularSupplyChainDT(episode_length=episode_length, seed=seed)
    env.reset(seed=seed)
    rows = []
    done = False
    while not done:
        _, metrics, done = env.step(noop_actions())
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    total_production = frame["production_input"].sum()
    total_energy = frame["energy_used"].sum()
    episode_kpis = measure_episode_kpis(env, rows)
    return {
        "seed": seed,
        "average_lead_time": episode_kpis["average_lead_time"],
        "service_level": float(frame.iloc[-1]["service_level"]),
        "otif": float(frame.iloc[-1]["otif"]),
        "operating_cost": float(frame["operating_cost"].sum()),
        "cost_per_unit": np.nan,
        "cost_per_delivered_unit": episode_kpis["cost_per_delivered_unit"],
        "co2_emissions": float(frame["emissions"].sum()),
        "energy_per_unit": float(total_energy / total_production),
        "material_waste": float(frame["net_waste"].sum()),
        "average_inventory": float(frame["inventory"].mean()),
        "wip": np.nan,
        "backlog": float(frame["backlog"].mean()),
        "completed_order_count": episode_kpis["completed_order_count"],
        "censored_order_count": episode_kpis["censored_order_count"],
    }


def collect_noop_references(seeds, output_dir, episode_length=None):
    """Collect per-seed values and provisional descriptive statistics.

    No statistic is selected as a reward normalizer by this function. The
    optional selector below makes that choice explicit and labels it as a
    proposal.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one fixed seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique for the reference rollout.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame([
        run_noop_episode(seed, episode_length) for seed in seeds
    ])
    per_seed.to_csv(output_path / "noop_kpi_per_seed.csv", index=False)

    summary_rows = []
    for metric in KPI_DEFINITIONS:
        values = per_seed[metric].dropna().to_numpy(dtype=float)
        available = metric not in UNAVAILABLE_KPIS and len(values) > 0
        if metric in NON_NORMALIZER_METRICS:
            normalizer_status = "proxy/diagnostic only; not a paper KPI normalizer"
        else:
            normalizer_status = "candidate/provisional; not selected"
        summary_rows.append({
            "metric": metric,
            "status": "available" if available else "unavailable",
            "definition": KPI_DEFINITIONS[metric],
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "minimum": float(np.min(values)) if len(values) else np.nan,
            "maximum": float(np.max(values)) if len(values) else np.nan,
            "sample_count": int(len(values)),
            "normalizer_status": normalizer_status,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path / "noop_kpi_summary.csv", index=False)
    pd.DataFrame([
        {"key": "episode_length", "value": episode_length or paper_comparable_config().episode_horizon},
        {"key": "seeds", "value": ",".join(str(seed) for seed in seeds)},
        {"key": "normalizer_policy", "value": "descriptive only; no automatic selection"},
    ]).to_csv(output_path / "noop_kpi_metadata.csv", index=False)
    return per_seed, summary


def select_candidate_normalizers(summary, statistic="mean"):
    """Explicitly propose fixed normalizers for one run; never paper-verified."""
    if statistic not in {"mean", "minimum", "maximum"}:
        raise ValueError("statistic must be mean, minimum, or maximum")
    selected = {}
    for row in summary.itertuples(index=False):
        if row.status != "available" or row.metric in NON_NORMALIZER_METRICS:
            continue
        value = float(getattr(row, statistic))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Cannot select non-positive normalizer for {row.metric}")
        selected[row.metric] = value
    return {
        "status": "candidate/provisional",
        "statistic": statistic,
        "values": selected,
        "fixed_within_evaluation_run": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="marl_artifacts/noop_normalization")
    args = parser.parse_args()
    _, summary = collect_noop_references(args.seeds, args.output_dir, 20)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()