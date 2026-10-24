"""Linear cooperative agents for the paper-inspired environment.

The paper identifies linear DQN-style learning and CTDE, but does not expose
machine-readable local-observation tables in the available PDF. The
observation definitions in this module are therefore explicit implementation
assumptions, while the action meanings remain those of paper_environment.py.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


AGENT_NAMES = (
    "planning",
    "inventory",
    "logistics",
    "expediting",
    "recycling",
)


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


def _value(source, key, default=0.0):
    return float(source.get(key, default))


def build_observation(agent_name, state, metrics=None, episode_length=60):
    """Build a bounded local observation for one specialized agent.

    Local features use the global digital-twin state plus the previous-step
    metrics available at decision time. Scaling constants are assumptions
    chosen to keep the linear features numerically well-conditioned.
    """
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"Unknown agent: {agent_name}")
    metrics = metrics or {}
    day = _value(state, "day") / max(float(episode_length), 1.0)
    inventory = _value(state, "inventory") / 1500.0
    backlog = _value(state, "backlog") / 1500.0
    recovered = _value(state, "recovered_material") / 80.0
    shipments = _value(state, "pending_shipments") / 10.0
    demand_uplift = _value(state, "demand_uplift")
    transport_delay = _value(state, "transport_delay_multiplier")
    energy_multiplier = _value(state, "energy_cost_multiplier")
    demand = _value(metrics, "demand") / 1000.0
    transit = _value(metrics, "transit_days") / 5.0
    service = _value(metrics, "service_level")
    otif = _value(metrics, "otif")
    waste = _value(metrics, "net_waste") / 100.0
    energy = _value(metrics, "energy_used") / 1000.0

    observations = {
        "planning": [day, inventory, backlog, demand, recovered, energy],
        "inventory": [day, inventory, backlog, demand, shipments, service],
        "logistics": [day, inventory, backlog, shipments, transit, transport_delay],
        "expediting": [day, backlog, shipments, transit, service, otif],
        "recycling": [day, recovered, waste, inventory, energy, energy_multiplier],
    }
    values = np.asarray(observations[agent_name], dtype=np.float64)
    return np.concatenate(([1.0], values))


class LinearDQNAgent:
    """A small replay-trained linear action-value learner.

    The paper's exact replay size, batch size, optimizer, and target-network
    schedule are not legible in the local PDF. This implementation uses
    uniform replay and a one-step semi-gradient Q update as documented
    assumptions.
    """

    def __init__(
        self,
        name,
        observation_size,
        action_count=3,
        seed=0,
        learning_rate=0.01,
        discount_factor=0.95,
        replay_capacity=2000,
        batch_size=32,
        epsilon=1.0,
        epsilon_decay=0.995,
        minimum_epsilon=0.05,
    ):
        if name not in AGENT_NAMES:
            raise ValueError(f"Unknown agent: {name}")
        self.name = name
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((action_count, observation_size), dtype=np.float64)
        self.replay = deque(maxlen=replay_capacity)

    def q_values(self, observation):
        observation = np.asarray(observation, dtype=np.float64)
        values = self.weights @ observation
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Non-finite Q-values for {self.name}")
        return values

    def select_action(self, observation, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        return int(np.argmax(self.q_values(observation)))

    def remember(self, observation, action, reward, next_observation, done):
        if action not in range(self.action_count):
            raise ValueError(f"Invalid action {action} for {self.name}")
        self.replay.append(Transition(
            np.asarray(observation, dtype=np.float64),
            int(action),
            float(reward),
            np.asarray(next_observation, dtype=np.float64),
            bool(done),
        ))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        indices = self.rng.choice(len(self.replay), self.batch_size, replace=False)
        losses = []
        for index in indices:
            transition = self.replay[index]
            current = self.q_values(transition.observation)[transition.action]
            future = 0.0 if transition.done else np.max(
                self.q_values(transition.next_observation)
            )
            target = transition.reward + self.discount_factor * future
            error = target - current
            self.weights[transition.action] += (
                self.learning_rate * error * transition.observation
            )
            losses.append(error * error)
        return float(np.mean(losses))

    def decay_exploration(self):
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            name=self.name,
            weights=self.weights,
            epsilon=np.asarray(self.epsilon),
            learning_rate=np.asarray(self.learning_rate),
            discount_factor=np.asarray(self.discount_factor),
        )
