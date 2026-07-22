"""DQN action selection: epsilon-greedy schedules and R-Max optimistic policies.

Extracted from ``dqn.training``. Re-exported by ``dqn.training`` for backwards
compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rlflow_builtin.dqn.config import DqnAgentConfig, DqnIntrinsicConfig
from rlflow_builtin.dqn.intrinsic import (
    _count_direct_bonus,
    _count_direct_bonus_for_all_actions,
    _count_uses_oracle_tabular,
    _intrinsic_bonus,
    _intrinsic_bonus_for_all_actions,
)
from rlflow_builtin.dqn.networks import _apply_mlp
from rlflow_builtin.dqn.state import DqnIntrinsicState


def _epsilon(
    step: jax.Array,
    start: float,
    end: float,
    decay_steps: int,
) -> jax.Array:
    # Guard against decay_steps == 0, which would divide by zero (NaN at step 0).
    decay = float(max(int(decay_steps), 1))
    fraction = jnp.minimum(1.0, step.astype(jnp.float32) / decay)
    return start + fraction * (end - start)


def _epsilon_greedy_action(
    greedy_action: jax.Array,
    random_key: jax.Array,
    choice_key: jax.Array,
    epsilon: jax.Array,
    num_actions: int,
) -> jax.Array:
    random_action = jax.random.randint(random_key, (), 0, num_actions, dtype=jnp.int32)
    explore = jax.random.uniform(choice_key) < epsilon
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def _select_action(
    agent: DqnAgentConfig,
    params,
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    observation: jax.Array,
    key: jax.Array,
    num_actions: int,
    global_step: jax.Array,
    *,
    training: bool,
    state_id: jax.Array | None = None,
) -> jax.Array:
    epsilon = (
        _epsilon(
            global_step,
            agent.epsilon_start,
            agent.epsilon_end,
            agent.epsilon_decay_steps,
        )
        if training
        else jnp.asarray(agent.eval_epsilon, dtype=jnp.float32)
    )
    greedy_key, random_key, choice_key = jax.random.split(key, 3)
    if agent.algorithm == "dqn_rmax":
        greedy_action = _rmax_action(
            agent,
            params,
            intrinsic_state,
            intrinsic,
            observation,
            greedy_key,
            num_actions,
            state_id=state_id,
        )
        return _epsilon_greedy_action(
            greedy_action,
            random_key,
            choice_key,
            epsilon,
            num_actions,
        )
    q_values = _apply_mlp(params, observation[None, :], agent.activation).squeeze(0)
    greedy_action = _max_action(q_values, greedy_key)
    return _epsilon_greedy_action(
        greedy_action,
        random_key,
        choice_key,
        epsilon,
        num_actions,
    )


def _rmax_action(
    agent: DqnAgentConfig,
    params,
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    observation: jax.Array,
    key: jax.Array,
    num_actions: int,
    state_id: jax.Array | None = None,
) -> jax.Array:
    q_values = _apply_mlp(params, observation[None, :], agent.activation).squeeze(0)
    if _count_uses_oracle_tabular(intrinsic):
        if state_id is None:
            raise ValueError("oracle_tabular count requires state_id for R-Max action selection")
        bonuses = _count_direct_bonus_for_all_actions(
            intrinsic_state,
            state_id.reshape((1,)),
            intrinsic,
            num_actions,
        ).squeeze(0)
    else:
        bonuses = _intrinsic_bonus_for_all_actions(
            intrinsic_state,
            observation[None, :],
            intrinsic,
            num_actions,
        ).squeeze(0)
    optimistic_values = jnp.where(
        bonuses > agent.rmax_bonus_threshold,
        agent.rmax_decision_v_max,
        q_values,
    )
    return _max_action(optimistic_values, key)


def _max_action(values: jax.Array, key: jax.Array) -> jax.Array:
    max_value = jnp.max(values)
    ties = values == max_value
    tie_count = jnp.sum(ties.astype(jnp.int32))
    selected_tie = jax.random.randint(key, (), 0, tie_count, dtype=jnp.int32)
    tie_offsets = jnp.cumsum(ties.astype(jnp.int32)) - 1
    return jnp.argmax(jnp.logical_and(ties, tie_offsets == selected_tie)).astype(jnp.int32)


def _max_actions(values: jax.Array, key: jax.Array) -> jax.Array:
    keys = jax.random.split(key, values.shape[0])
    return jax.vmap(_max_action)(values, keys).astype(jnp.int32)


def _rmax_batch_masks(
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    agent: DqnAgentConfig,
    batch: dict[str, jax.Array],
    num_actions: int,
) -> tuple[jax.Array, jax.Array]:
    actions = batch["actions"]
    if agent.algorithm != "dqn_rmax":
        return (
            jnp.ones_like(actions, dtype=jnp.bool_),
            jnp.zeros_like(actions, dtype=jnp.bool_),
        )
    if _count_uses_oracle_tabular(intrinsic):
        current_bonus = _count_direct_bonus(
            intrinsic_state,
            batch["state_ids"],
            actions,
            intrinsic,
            num_actions,
        )
        next_bonuses = _count_direct_bonus_for_all_actions(
            intrinsic_state,
            batch["next_state_ids"],
            intrinsic,
            num_actions,
        )
    else:
        current_bonus = _intrinsic_bonus(
            intrinsic_state,
            batch["observations"],
            actions,
            intrinsic,
            num_actions,
        )
        next_bonuses = _intrinsic_bonus_for_all_actions(
            intrinsic_state,
            batch["next_observations"],
            intrinsic,
            num_actions,
        )
    return (
        current_bonus <= agent.rmax_bonus_threshold,
        jnp.any(next_bonuses > agent.rmax_bonus_threshold, axis=1),
    )
