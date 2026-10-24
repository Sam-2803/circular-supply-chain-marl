"""Configurable, paper-aligned observation features.

The paper supports normalized KPI snapshots, operational context, and action
encoding, but the exact per-agent tables are unresolved. This module therefore
offers explicit information modes without changing the existing observation
function or training path.
"""

from dataclasses import dataclass

import numpy as np

from marl_agents import AGENT_NAMES


KPI_ORDER = (
    "service_level",
    "otif",
    "operating_cost",
    "co2_emissions",
    "energy_per_unit",
    "material_waste",
    "average_inventory",
    "backlog",
    "average_lead_time",
    "wip",
)

KPI_SOURCE_NAMES = {
    "co2_emissions": "emissions",
    "material_waste": "net_waste",
    "average_inventory": "inventory",
}

FULL_DATA_KPIS = KPI_ORDER

# These sets are provisional silo access proposals, not paper-verified tables.
SILO_DATA_KPIS = {
    "planning": (
        "service_level", "otif", "average_inventory", "backlog", "wip"
    ),
    "inventory": (
        "service_level", "otif", "average_inventory", "backlog", "wip"
    ),
    "logistics": (
        "service_level", "otif", "average_lead_time", "backlog", "wip"
    ),
    "expediting": (
        "service_level", "otif", "average_lead_time", "backlog"
    ),
    "recycling": (
        "co2_emissions", "energy_per_unit", "material_waste", "wip"
    ),
}

CONTEXT_ORDER = (
    "day",
    "pending_shipments",
    "recovered_material",
    "production_capacity",
    "transport_capacity",
    "recycling_capacity",
    "demand_uplift",
    "transport_delay_multiplier",
    "energy_cost_multiplier",
)

CONTEXT_SCALES = {
    "day": 20.0,
    "pending_shipments": 10.0,
    "recovered_material": 80.0,
    "production_capacity": 600.0,
    "transport_capacity": 600.0,
    "recycling_capacity": 80.0,
    "demand_uplift": 1.0,
    "transport_delay_multiplier": 2.0,
    "energy_cost_multiplier": 2.0,
}


class MissingObservationMetricError(ValueError):
    """Raised when strict observation construction encounters missing data."""


@dataclass(frozen=True)
class NormalizerConfig:
    """Fixed normalizers supplied by the caller for one run.

    Values are intentionally not inferred from candidate No-Op CSV output.
    """

    values: dict[str, float]
    label: str = "caller-supplied"

    def validate(self):
        for name, value in self.values.items():
            if name not in KPI_ORDER:
                raise ValueError(f"Unknown KPI normalizer: {name}")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Normalizer for {name} must be positive and finite")


@dataclass(frozen=True)
class ObservationResult:
    values: np.ndarray
    feature_names: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    mode: str


def _source(metrics, state, name):
    source_name = KPI_SOURCE_NAMES.get(name, name)
    if source_name in metrics:
        return float(metrics[source_name])
    if source_name in state:
        return float(state[source_name])
    return None


def _energy_per_unit(metrics):
    energy = metrics.get("energy_used")
    production = metrics.get("production_input")
    if energy is None or production is None or float(production) <= 0:
        return None
    return float(energy) / float(production)


def _kpi_values(state, metrics):
    values = {}
    for name in KPI_ORDER:
        if name == "energy_per_unit":
            value = _energy_per_unit(metrics)
        else:
            value = _source(metrics, state, name)
        if value is not None:
            values[name] = value
    return values


def _allowed_kpis(agent_name, mode):
    if mode == "full_data":
        return FULL_DATA_KPIS
    if mode == "silo_data":
        return SILO_DATA_KPIS[agent_name]
    if mode == "exploratory":
        return ()
    raise ValueError("mode must be exploratory, full_data, or silo_data")


def build_observation(
    agent_name,
    state,
    metrics=None,
    normalizers=None,
    mode="full_data",
):
    """Build paper-aligned normalized features without changing environment state.

    Missing KPIs are omitted from the feature vector and returned explicitly.
    The exact silo access sets are provisional and configurable by selecting
    ``full_data`` or ``silo_data``; no mode is silently treated as paper exact.
    """
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"Unknown agent: {agent_name}")
    metrics = metrics or {}
    normalizers = normalizers or NormalizerConfig({})
    normalizers.validate()
    allowed = _allowed_kpis(agent_name, mode)
    values = _kpi_values(state, metrics)
    missing = tuple(
        name for name in allowed
        if name not in values or name not in normalizers.values
    )

    feature_values = [1.0]
    feature_names = ["bias"]
    for name in allowed:
        if name in missing:
            continue
        feature_values.append(values[name] / normalizers.values[name])
        feature_names.append(f"kpi:{name}")

    for name in CONTEXT_ORDER:
        if name in state:
            feature_values.append(float(state[name]) / CONTEXT_SCALES[name])
            feature_names.append(f"context:{name}")

    result = ObservationResult(
        values=np.asarray(feature_values, dtype=np.float64),
        feature_names=tuple(feature_names),
        missing_metrics=missing,
        mode=mode,
    )
    if not np.isfinite(result.values).all():
        raise FloatingPointError("Observation contains non-finite values")
    return result


def action_feature_map(observation, action, action_count=3):
    """Append one-hot action encoding to an observation for linear Q features."""
    if action not in range(action_count):
        raise ValueError(f"Action must be in 0..{action_count - 1}")
    vector = np.asarray(observation.values, dtype=np.float64)
    one_hot = np.zeros(action_count, dtype=np.float64)
    one_hot[action] = 1.0
    names = observation.feature_names + tuple(
        f"action:{index}" for index in range(action_count)
    )
    return ObservationResult(
        values=np.concatenate((vector, one_hot)),
        feature_names=names,
        missing_metrics=observation.missing_metrics,
        mode=observation.mode,
    )