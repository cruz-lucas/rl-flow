"""DQN state and result containers (JAX NamedTuples + the run result/env types).

Extracted from ``dqn.training`` so the replay/intrinsic/policy modules can share
these container types without importing the training loop. Re-exported by
``dqn.training`` for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import numpy as np
import optax


@dataclass(frozen=True)
class DqnRunResult:
    params: tuple[dict[str, jax.Array], ...]
    aux_params: dict[str, tuple[dict[str, jax.Array], ...]]
    train_returns: np.ndarray
    train_discounted_returns: np.ndarray | None
    train_lengths: np.ndarray
    train_losses: np.ndarray
    eval_returns: np.ndarray
    eval_discounted_returns: np.ndarray | None
    eval_lengths: np.ndarray
    source_observation_shape: tuple[int, ...]
    source_observation_dtype: str
    input_dim: int
    num_actions: int
    replay_arrays: dict[str, np.ndarray] | None = None
    count_table_entries: int | None = None
    count_table_overflow: bool | None = None


class DqnReplayState(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    intrinsic_targets: jax.Array
    reward_intrinsic_targets: jax.Array
    state_ids: jax.Array
    next_state_ids: jax.Array
    source_observations: jax.Array
    source_next_observations: jax.Array
    size: jax.Array
    index: jax.Array


class DqnIntrinsicState(NamedTuple):
    target_params: tuple[dict[str, jax.Array], ...]
    prior_params: tuple[dict[str, jax.Array], ...]
    predictor_params: tuple[dict[str, jax.Array], ...]
    opt_state: optax.OptState
    reward_mean: jax.Array
    reward_var: jax.Array
    count_keys: jax.Array
    counts: jax.Array
    count_size: jax.Array
    count_overflow: jax.Array


class DqnTrainState(NamedTuple):
    params: tuple[dict[str, jax.Array], ...]
    target_params: tuple[dict[str, jax.Array], ...]
    opt_state: optax.OptState
    intrinsic_state: DqnIntrinsicState
    reward_intrinsic_state: DqnIntrinsicState
    replay_state: DqnReplayState
    key: jax.Array
    global_step: jax.Array
    gradient_step: jax.Array
    intrinsic_gradient_step: jax.Array
    reward_intrinsic_gradient_step: jax.Array


@dataclass(frozen=True)
class _DqnEnvironment:
    observation_shape: tuple[int, ...]
    observation_dtype: str
    input_dim: int
    num_actions: int
    reset: Callable[[jax.Array], Any]
    step: Callable[[Any, jax.Array, jax.Array], Any]
    observation: Callable[[Any], Any]
    reward: Callable[[Any], jax.Array]
    done: Callable[[Any], jax.Array]
    encode: Callable[[Any], jax.Array]
    oracle_state_id: Callable[[Any], jax.Array] | None = None
    oracle_state_space_size: int | None = None
