from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import numpy as np


AlgorithmName = Literal["q_learning", "sarsa"]
PolicyName = Literal["epsilon_greedy", "ucb", "softmax"]
EnvironmentName = Literal["gridworld", "riverswim", "sixarms", "navix"]
BufferName = Literal["none", "uniform"]


@dataclass(frozen=True)
class AgentConfig:
    algorithm: AlgorithmName
    learning_rate: float
    discount: float
    initial_q: float


@dataclass(frozen=True)
class PolicyConfig:
    name: PolicyName
    epsilon: float = 0.1
    eval_epsilon: float = 0.0
    coefficient: float = 1.0
    initial_count: float = 1.0
    temperature: float = 1.0
    eval_temperature: float = 0.01


@dataclass(frozen=True)
class RunnerConfig:
    seed: int
    train_episodes: int
    train_steps: int | None
    max_episode_steps: int
    eval_episodes: int
    checkpoint_freq: int | None
    checkpoint_dir: str
    save_final_checkpoint: bool


@dataclass(frozen=True)
class EnvironmentConfig:
    name: EnvironmentName
    num_states: int
    num_actions: int
    start_state: int
    random_start_states: tuple[int, ...] = ()
    width: int = 0
    height: int = 0
    goal_state: int = 0
    pit_states: tuple[int, ...] = ()
    goal_reward: float = 1.0
    pit_reward: float = -1.0
    step_reward: float = 0.0
    slip_probability: float = 0.0
    p_right: float = 0.3
    p_left: float = 0.1
    p_stay: float = 0.6
    hard_reward: float = 10_000.0
    easy_reward: float = 5.0
    success_probabilities: tuple[float, ...] = ()
    arm_rewards: tuple[float, ...] = ()
    navix_env_name: str = "empty_room"
    navix_size: int = 5
    navix_layout: str = "fixed"
    navix_observation_mode: str = "tabular"
    navix_action_set: str = "default"
    navix_max_steps: int | None = None


@dataclass(frozen=True)
class BufferConfig:
    name: BufferName = "none"
    capacity: int = 1
    batch_size: int = 1
    min_size: int = 1
    updates_per_step: int = 0
    save_dataset_path: str = ""
    load_dataset_path: str = ""
    offline_only: bool = False
    offline_updates: int = 0

    @property
    def enabled(self) -> bool:
        return self.name != "none" or bool(self.load_dataset_path)


class ReplayBufferState(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    size: jax.Array
    index: jax.Array


class TransitionBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array


@dataclass(frozen=True)
class TabularDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray


@dataclass(frozen=True)
class TabularRunResult:
    q_table: np.ndarray
    action_counts: np.ndarray
    train_returns: np.ndarray
    train_lengths: np.ndarray
    train_losses: np.ndarray
    eval_returns: np.ndarray
    eval_lengths: np.ndarray
    dataset: TabularDataset | None = None
