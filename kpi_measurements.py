"""Defensible diagnostic KPI measurements for completed episodes."""

import numpy as np


def measure_episode_kpis(environment, daily_metrics):
    """Return exact, proxy, and unavailable KPI measurements.

    Lead time is computed only for fully completed orders. Delivered-unit
    cost is an explicit operational proxy, not the paper's unresolved metric.
    WIP is unavailable because production-stage work is not tracked.
    """
    rows = list(daily_metrics)
    completed_orders = [
        order for order in environment.orders
        if order["completed_day"] is not None
    ]
    lead_times = [
        order["completed_day"] - order["created_day"]
        for order in completed_orders
    ]
    total_cost = float(sum(row["operating_cost"] for row in rows))
    total_delivered = float(sum(row["delivered"] for row in rows))
    return {
        "average_lead_time": (
            float(np.mean(lead_times)) if lead_times else np.nan
        ),
        "average_lead_time_unit": "days",
        "completed_order_count": len(completed_orders),
        "censored_order_count": len(environment.orders) - len(completed_orders),
        "cost_per_delivered_unit": (
            total_cost / total_delivered if total_delivered > 0 else np.nan
        ),
        "cost_per_delivered_unit_status": "proxy; not paper-defined cost per unit",
        "wip": np.nan,
        "wip_status": "unavailable; production-stage WIP is not tracked",
    }