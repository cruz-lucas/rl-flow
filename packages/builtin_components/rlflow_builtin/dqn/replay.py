"""DQN replay-buffer I/O: init, push, sample, and export helpers.

Extracted from ``dqn.training``; depends only on the state/config containers.
Re-exported by ``dqn.training`` for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.dqn.config import DqnIntrinsicConfig
from rlflow_builtin.dqn.state import DqnReplayState


def _initial_replay_state(
    capacity: int,
    input_dim: int,
    intrinsic_target_dim: int,
    source_observation_shape: tuple[int, ...],
    source_observation_dtype: str,
    reward_intrinsic_target_dim: int | None = None,
) -> DqnReplayState:
    source_shape = (capacity, *source_observation_shape)
    source_dtype = np.dtype(source_observation_dtype)
    reward_intrinsic_target_dim = reward_intrinsic_target_dim or 1
    return DqnReplayState(
        observations=jnp.zeros((capacity, input_dim), dtype=jnp.float32),
        actions=jnp.zeros((capacity,), dtype=jnp.int32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_observations=jnp.zeros((capacity, input_dim), dtype=jnp.float32),
        terminals=jnp.zeros((capacity,), dtype=jnp.float32),
        intrinsic_targets=jnp.zeros((capacity, intrinsic_target_dim), dtype=jnp.float32),
        reward_intrinsic_targets=jnp.zeros(
            (capacity, reward_intrinsic_target_dim),
            dtype=jnp.float32,
        ),
        state_ids=jnp.zeros((capacity,), dtype=jnp.int32),
        next_state_ids=jnp.zeros((capacity,), dtype=jnp.int32),
        source_observations=jnp.zeros(source_shape, dtype=source_dtype),
        source_next_observations=jnp.zeros(source_shape, dtype=source_dtype),
        size=jnp.asarray(0, dtype=jnp.int32),
        index=jnp.asarray(0, dtype=jnp.int32),
    )


def _push_replay(
    state: DqnReplayState,
    observation: jax.Array,
    source_observation: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_observation: jax.Array,
    source_next_observation: jax.Array,
    terminal: jax.Array,
    intrinsic_target: jax.Array,
    state_id: jax.Array,
    next_state_id: jax.Array,
    reward_intrinsic_target: jax.Array | None = None,
) -> DqnReplayState:
    index = state.index
    capacity = state.observations.shape[0]
    if reward_intrinsic_target is None:
        reward_intrinsic_target = jnp.zeros(
            (state.reward_intrinsic_targets.shape[-1],),
            dtype=jnp.float32,
        )
    return DqnReplayState(
        observations=state.observations.at[index].set(observation),
        actions=state.actions.at[index].set(action.astype(jnp.int32)),
        rewards=state.rewards.at[index].set(reward.astype(jnp.float32)),
        next_observations=state.next_observations.at[index].set(next_observation),
        terminals=state.terminals.at[index].set(terminal.astype(jnp.float32)),
        intrinsic_targets=state.intrinsic_targets.at[index].set(intrinsic_target),
        reward_intrinsic_targets=state.reward_intrinsic_targets.at[index].set(
            reward_intrinsic_target
        ),
        state_ids=state.state_ids.at[index].set(state_id.astype(jnp.int32)),
        next_state_ids=state.next_state_ids.at[index].set(next_state_id.astype(jnp.int32)),
        source_observations=state.source_observations.at[index].set(
            source_observation.astype(state.source_observations.dtype)
        ),
        source_next_observations=state.source_next_observations.at[index].set(
            source_next_observation.astype(state.source_next_observations.dtype)
        ),
        size=jnp.minimum(state.size + 1, capacity).astype(jnp.int32),
        index=((index + 1) % capacity).astype(jnp.int32),
    )


def _sample_replay(
    state: DqnReplayState,
    key: jax.Array,
    batch_size: int,
) -> dict[str, jax.Array]:
    indices = jax.random.randint(key, (batch_size,), 0, state.size, dtype=jnp.int32)
    return {
        "observations": state.observations[indices],
        "actions": state.actions[indices],
        "rewards": state.rewards[indices],
        "next_observations": state.next_observations[indices],
        "terminals": state.terminals[indices],
        "intrinsic_targets": state.intrinsic_targets[indices],
        "reward_intrinsic_targets": state.reward_intrinsic_targets[indices],
        "state_ids": state.state_ids[indices],
        "next_state_ids": state.next_state_ids[indices],
    }


def _replay_intrinsic_target_dim(intrinsic: DqnIntrinsicConfig) -> int:
    if intrinsic.kind == "cfn":
        return intrinsic.output_dim
    return 1


def _replay_arrays(
    state: DqnReplayState,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    *,
    shared_intrinsic: bool,
) -> dict[str, np.ndarray]:
    size = int(np.asarray(jax.device_get(state.size)))
    arrays = {
        "observations": np.asarray(jax.device_get(state.source_observations))[:size],
        "actions": np.asarray(jax.device_get(state.actions))[:size],
        "rewards": np.asarray(jax.device_get(state.rewards))[:size],
        "next_observations": np.asarray(jax.device_get(state.source_next_observations))[:size],
        "terminals": np.asarray(jax.device_get(state.terminals))[:size].astype(np.bool_),
    }
    if shared_intrinsic and knownness.kind == "cfn":
        arrays["cfn_targets"] = np.asarray(jax.device_get(state.intrinsic_targets))[:size]
    elif knownness.kind == "cfn" and intrinsic_reward.kind == "none":
        arrays["cfn_targets"] = np.asarray(jax.device_get(state.intrinsic_targets))[:size]
    elif intrinsic_reward.kind == "cfn" and knownness.kind == "none":
        arrays["cfn_targets"] = np.asarray(jax.device_get(state.reward_intrinsic_targets))[:size]
    else:
        if knownness.kind == "cfn":
            arrays["knownness_cfn_targets"] = np.asarray(jax.device_get(state.intrinsic_targets))[
                :size
            ]
        if intrinsic_reward.kind == "cfn":
            arrays["intrinsic_reward_cfn_targets"] = np.asarray(
                jax.device_get(state.reward_intrinsic_targets)
            )[:size]
    return arrays


def _resolve_replay_save_path(path: str, run_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    if candidate.is_absolute():
        return candidate
    return (run_dir / candidate).resolve()
