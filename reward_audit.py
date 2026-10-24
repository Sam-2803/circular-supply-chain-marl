"""Read-only comparison metadata for the paper and exploratory reward."""

from paper_config import paper_comparable_config


PAPER_REWARD_EQUATION = (
    "r_t = -sum_k(omega_k * normalized_KPI_k,t) + b_t"
)
PAPER_NORMALIZATION_EQUATION = (
    "normalized_KPI_k,t = KPI_k,t / K_k^norm"
)

EXPLORATORY_REWARD_TERMS = {
    "co2": {"weight": 1.1, "normalizer": 5000.0, "orientation": "lower is better"},
    "cost": {"weight": 1.2, "normalizer": 15000.0, "orientation": "lower is better"},
    "inventory": {"weight": 0.5, "normalizer": 600.0, "orientation": "lower is better as a holding proxy"},
    "lead_time": {"weight": 0.7, "normalizer": 3.0, "orientation": "lower is better"},
    "waste": {"weight": 1.2, "normalizer": 20.0, "orientation": "lower is better"},
    "backlog": {"weight": 0.6, "normalizer": 1500.0, "orientation": "lower is better"},
    "cost_per_unit": {"weight": 1.0, "normalizer": 30.0, "orientation": "lower is better"},
    "energy_per_unit": {"weight": 0.6, "normalizer": 2.0, "orientation": "lower is better"},
}

EXPLORATORY_BONUSES = {
    "service_level_threshold": {"threshold": 0.95, "bonus": 0.5},
    "otif_threshold": {"threshold": 0.95, "bonus": 0.5},
    "recovery_ratio_threshold": {"threshold": 0.4, "bonus": 0.25},
}

UNRESOLVED_REWARD_DETAILS = (
    "paper KPI weights",
    "paper No-Op normalizer aggregation and units",
    "paper additive term b_t",
    "paper orientation transform for benefit KPIs",
    "paper clipping rules",
    "paper threshold or terminal bonuses",
)


def reward_audit_report():
    """Return structured evidence without calculating or changing rewards."""
    config = paper_comparable_config()
    return {
        "paper": {
            "page": 9,
            "equation": PAPER_REWARD_EQUATION,
            "normalization": PAPER_NORMALIZATION_EQUATION,
            "shared_reward": config.shared_global_reward,
            "weights": "unresolved",
            "normalizers": "positive, No-Op-derived, exact aggregation unresolved",
            "additive_term": "unresolved b_t",
            "clipping": "unresolved",
            "bonuses": "unresolved",
            "terminal_adjustments": "unresolved",
        },
        "exploratory": {
            "terms": EXPLORATORY_REWARD_TERMS,
            "clipping": "each normalized term clipped at 2.0",
            "bonuses": EXPLORATORY_BONUSES,
            "terminal_adjustments": "none beyond terminal TD handling",
            "status": "provisional; not paper-faithful",
        },
        "unresolved": UNRESOLVED_REWARD_DETAILS,
    }


def require_paper_reward_details():
    """Fail explicitly because the available evidence is incomplete."""
    raise ValueError(
        "Paper-faithful reward construction is unavailable; unresolved details: "
        + ", ".join(UNRESOLVED_REWARD_DETAILS)
    )