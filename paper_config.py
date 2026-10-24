"""Configuration metadata for exploratory and paper-comparable experiments.

This module does not change the existing training behavior. It records which
details are supported by the readable paper evidence and prevents an
incomplete configuration from being labelled paper-faithful.
"""

from dataclasses import dataclass


AGENT_NAMES = (
    "planning",
    "inventory",
    "logistics",
    "expediting",
    "recycling",
)


class UnresolvedPaperSettingError(ValueError):
    """Raised when unresolved paper details are required for fidelity."""


@dataclass(frozen=True)
class PaperConfig:
    mode: str
    agent_names: tuple[str, ...]
    shared_global_reward: bool
    linear_action_value: bool
    fixed_feature_map: bool
    one_step_td_target: bool
    terminal_td_handling: bool
    greedy_evaluation: bool
    episode_horizon: int
    learning_rate_range: tuple[float, float] | None
    learning_rate_range_is_approximate: bool
    learning_rate: float | None
    reward_weights: object | None
    noop_normalizer_method: str | None
    additive_reward_term: object | None
    observation_features: object | None
    action_values: object | None
    discount_factor: float | None
    replay_settings: object | None
    epsilon_schedule: object | None

    @property
    def unresolved_settings(self) -> tuple[str, ...]:
        fields = (
            "reward_weights",
            "noop_normalizer_method",
            "additive_reward_term",
            "observation_features",
            "action_values",
            "discount_factor",
            "replay_settings",
            "epsilon_schedule",
        )
        return tuple(field for field in fields if getattr(self, field) is None)

    def require_paper_faithful(self) -> None:
        """Reject configurations that cannot support a paper-faithful claim."""
        if self.unresolved_settings:
            unresolved = ", ".join(self.unresolved_settings)
            raise UnresolvedPaperSettingError(
                "Paper-faithful configuration is unavailable; unresolved "
                f"settings: {unresolved}."
            )


def paper_comparable_config() -> PaperConfig:
    """Return verified metadata without inventing unresolved parameters."""
    return PaperConfig(
        mode="paper-comparable-incomplete",
        agent_names=AGENT_NAMES,
        shared_global_reward=True,
        linear_action_value=True,
        fixed_feature_map=True,
        one_step_td_target=True,
        terminal_td_handling=True,
        greedy_evaluation=True,
        episode_horizon=20,
        learning_rate_range=(3e-4, 3e-3),
        learning_rate_range_is_approximate=True,
        learning_rate=None,
        reward_weights=None,
        noop_normalizer_method=None,
        additive_reward_term=None,
        observation_features=None,
        action_values=None,
        discount_factor=None,
        replay_settings=None,
        epsilon_schedule=None,
    )


def exploratory_config() -> PaperConfig:
    """Return the existing provisional settings for explicitly exploratory runs."""
    return PaperConfig(
        mode="exploratory",
        agent_names=AGENT_NAMES,
        shared_global_reward=True,
        linear_action_value=True,
        fixed_feature_map=True,
        one_step_td_target=True,
        terminal_td_handling=True,
        greedy_evaluation=True,
        episode_horizon=60,
        learning_rate_range=None,
        learning_rate_range_is_approximate=False,
        learning_rate=0.01,
        reward_weights="provisional values in paper_environment.py",
        noop_normalizer_method="manual constants in paper_environment.py",
        additive_reward_term="provisional threshold/recovery bonuses",
        observation_features="provisional six-feature local observations",
        action_values="provisional environment action mappings",
        discount_factor=0.95,
        replay_settings={"capacity": 2000, "batch_size": 32},
        epsilon_schedule={
            "initial": 1.0,
            "decay": 0.995,
            "minimum": 0.05,
        },
    )