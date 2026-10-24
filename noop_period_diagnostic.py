"""Read-only No-Op seed-period panel for normalization audits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noop_normalization import noop_actions
from paper_config import paper_comparable_config
from paper_environment import CircularSupplyChainDT


SEEDS = tuple(range(100, 110))
HORIZON = 20

PERIOD_KPIS = (
    "operating_cost",
    "co2_emissions",
    "inventory",
    "material_waste",
    "backlog",
    "energy_per_unit",
    "service_level",
    "otif",
    "average_lead_time",
    "cost_per_unit",
    "wip",
)


def _ratio(numerator, denominator, reason):
    if denominator == 0:
        return np.nan, f"zero_denominator:{reason}"
    return float(numerator / denominator), "available"


def run_seed(seed, horizon=HORIZON):
    env = CircularSupplyChainDT(episode_length=horizon, seed=seed)
    env.reset(seed=seed)
    rows = []
    previous_on_time = 0.0
    previous_demand = 0.0
    previous_orders = 0
    previous_completed = 0
    previous_otif = 0

    for period in range(1, horizon + 1):
        _, metrics, done = env.step(noop_actions())
        on_time_delta = env.total_on_time_units - previous_on_time
        demand_delta = env.total_demand - previous_demand
        orders_delta = env.total_orders - previous_orders
        completed_delta = env.total_completed_orders - previous_completed
        otif_delta = env.total_otif_orders - previous_otif
        energy_per_unit, energy_status = _ratio(
            metrics["energy_used"], metrics["production_input"], "production_input"
        )
        service_period, service_status = _ratio(
            on_time_delta, demand_delta, "demand"
        )
        otif_period, otif_status = _ratio(
            otif_delta, orders_delta, "new_orders"
        )
        rows.append({
            "seed": seed,
            "period": period,
            "horizon": horizon,
            "scenario": "No-Op",
            "policy": "No-Op",
            "exogenous_demand": metrics["demand"],
            "production_input": metrics["production_input"],
            "usable_output": metrics["usable_output"],
            "yield_rate": metrics["yield_rate"],
            "operating_cost": metrics["operating_cost"],
            "co2_emissions": metrics["emissions"],
            "material_waste": metrics["net_waste"],
            "energy_used": metrics["energy_used"],
            "inventory_end": metrics["inventory"],
            "backlog_end": metrics["backlog"],
            "delivered_units": metrics["delivered"],
            "on_time_delivered_units": round(on_time_delta, 8),
            "demand_numerator": round(demand_delta, 8),
            "cumulative_demand": round(env.total_demand, 8),
            "cumulative_on_time_units": round(env.total_on_time_units, 8),
            "total_orders_to_date": env.total_orders,
            "completed_orders_to_date": env.total_completed_orders,
            "otif_orders_to_date": env.total_otif_orders,
            "new_orders": orders_delta,
            "completed_orders": completed_delta,
            "otif_orders": otif_delta,
            "pending_shipments": metrics["pending_shipments"],
            "baseline_transit_days": metrics["baseline_transit_days"],
            "energy_per_unit": energy_per_unit,
            "energy_per_unit_status": energy_status,
            "service_level_period": service_period,
            "service_level_period_status": service_status,
            "otif_period": otif_period,
            "otif_period_status": otif_status,
            "average_lead_time": np.nan,
            "average_lead_time_status": "order-level only; not period-defined",
            "cost_per_unit": np.nan,
            "cost_per_unit_status": "unavailable: paper denominator unresolved",
            "wip": np.nan,
            "wip_status": "unavailable: production WIP not tracked",
            "schedule_adherence": np.nan,
            "schedule_adherence_status": "unavailable: definition/measurement missing",
            "budget_favorable": np.nan,
            "budget_favorable_status": "unavailable: definition missing",
        })
        previous_on_time = env.total_on_time_units
        previous_demand = env.total_demand
        previous_orders = env.total_orders
        previous_completed = env.total_completed_orders
        previous_otif = env.total_otif_orders
        if done:
            break

    order_rows = []
    for order in env.orders:
        completed = order["completed_day"] is not None
        order_rows.append({
            "seed": seed,
            "order_id": order["id"],
            "created_day": order["created_day"],
            "due_day": order["due_day"],
            "completed_day": order["completed_day"],
            "delivered_quantity": order["delivered"],
            "required_quantity": order["quantity"],
            "completed_by_due": bool(
                completed and order["completed_day"] <= order["due_day"]
            ),
            "censored": not completed,
            "lead_time_days": (
                order["completed_day"] - order["created_day"]
                if completed else np.nan
            ),
        })
    return pd.DataFrame(rows), pd.DataFrame(order_rows)


def build_normalizer_summary(period_panel, order_panel):
    """Build explicit candidate summaries; values are never activated."""
    definitions = {
        "operating_cost": ("period_mean", "mean of valid period costs", "cost/period"),
        "co2_emissions": ("period_mean", "mean of valid period emissions", "emissions/period"),
        "inventory": ("period_mean", "mean of period-end inventory", "units"),
        "material_waste": ("period_mean", "mean of period net waste", "units/period"),
        "backlog": ("period_mean", "mean of period-end backlog", "units"),
        "energy_per_unit": ("ratio_of_totals", "sum energy_used / sum production_input", "energy/production unit"),
        "service_level": ("ratio_of_totals", "sum on_time_delivered_units / sum demand_numerator", "ratio"),
        "otif": ("ratio_of_totals", "sum otif_orders / sum new_orders", "ratio"),
        "average_lead_time": ("pooled_order_mean", "mean valid completed-order lead times", "days"),
        "cost_per_unit": ("unresolved", "paper denominator unavailable", "unresolved"),
        "wip": ("unresolved", "production WIP unavailable", "unresolved"),
    }
    rows = []
    period_field = {
        "inventory": "inventory_end",
        "backlog": "backlog_end",
    }
    for metric, (method, definition, units) in definitions.items():
        if metric == "operating_cost":
            values = period_panel[metric].to_numpy(dtype=float)
            value = float(np.mean(values))
            count = int(np.isfinite(values).sum())
        elif metric in {"co2_emissions", "inventory", "material_waste", "backlog"}:
            field = period_field.get(metric, metric)
            if field not in period_panel:
                field = metric
            values = period_panel[field].to_numpy(dtype=float)
            value = float(np.mean(values))
            count = int(np.isfinite(values).sum())
        elif metric == "energy_per_unit":
            value, status = _ratio(
                period_panel["energy_used"].sum(),
                period_panel["production_input"].sum(),
                "production_input",
            )
            count = int(period_panel["energy_per_unit"].notna().sum())
            if status != "available":
                method = "unresolved"
        elif metric == "service_level":
            value, status = _ratio(
                period_panel["on_time_delivered_units"].sum(),
                period_panel["demand_numerator"].sum(),
                "demand",
            )
            count = int(period_panel["service_level_period"].notna().sum())
            if status != "available":
                method = "unresolved"
        elif metric == "otif":
            value, status = _ratio(
                period_panel["otif_orders"].sum(),
                period_panel["new_orders"].sum(),
                "new_orders",
            )
            count = int(period_panel["otif_period"].notna().sum())
            if status != "available":
                method = "unresolved"
        elif metric == "average_lead_time":
            values = order_panel["lead_time_days"].dropna().to_numpy(dtype=float)
            value = float(np.mean(values)) if len(values) else np.nan
            count = int(len(values))
        else:
            value = np.nan
            count = 0
        valid = np.isfinite(value) and value > 0
        rows.append({
            "metric": metric,
            "candidate_normalizer": value if valid else np.nan,
            "status": "candidate/provisional" if valid else "unavailable_or_invalid",
            "aggregation_method": method,
            "definition": definition,
            "units": units,
            "valid_observation_count": count,
            "normalizers_activated": False,
        })
    return pd.DataFrame(rows)


def run_diagnostic(seeds=SEEDS, output_dir="marl_artifacts/noop_period_diagnostic_stage6b", horizon=HORIZON):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a non-empty list of unique seeds")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    panels = [run_seed(seed, horizon) for seed in seeds]
    period_panel = pd.concat([item[0] for item in panels], ignore_index=True)
    order_panel = pd.concat([item[1] for item in panels], ignore_index=True)
    if period_panel.duplicated(["seed", "period"]).any():
        raise AssertionError("Duplicate seed-period observations")
    period_panel.to_csv(output_path / "noop_period_panel.csv", index=False)
    order_panel.to_csv(output_path / "noop_order_panel.csv", index=False)
    summary = build_normalizer_summary(period_panel, order_panel)
    summary.to_csv(output_path / "candidate_normalizers.csv", index=False)
    metadata = {
        "seeds": list(seeds),
        "horizon": horizon,
        "policy": "No-Op",
        "row_count": len(period_panel),
        "expected_row_count": len(seeds) * horizon,
        "outputs": [
            "noop_period_panel.csv",
            "noop_order_panel.csv",
            "candidate_normalizers.csv",
            "diagnostic_metadata.json",
        ],
        "normalizer_policy": "candidate/provisional only; estimated once and not activated",
        "limitations": [
            "Cost per unit, WIP, schedule adherence, and budget-favorable status are unavailable.",
            "Lead time is order-level and excludes censored orders.",
            "Exact paper KPI definitions and ratio aggregation remain unresolved.",
        ],
    }
    (output_path / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return period_panel, order_panel, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", default="marl_artifacts/noop_period_diagnostic_stage6b")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    _, _, summary = run_diagnostic(args.seeds, args.output_dir, args.horizon)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()