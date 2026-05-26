from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.tabular.types import BufferConfig, ReplayBufferState, TransitionBatch


def no_buffer_config() -> BufferConfig:
    return BufferConfig()


def buffer_config(component_id: str, config: dict[str, Any]) -> BufferConfig:
    if component_id != "builtin.replay.tabular_uniform":
        raise ValueError(f"Unsupported builtin tabular replay buffer: {component_id}")

    capacity = int(config["capacity"])
    batch_size = int(config["batch_size"])
    min_size = int(config["min_size"])
    updates_per_step = int(config["updates_per_step"])
    if min_size > capacity:
        raise ValueError("builtin.replay.tabular_uniform min_size cannot exceed capacity")
    if batch_size > capacity:
        raise ValueError("builtin.replay.tabular_uniform batch_size cannot exceed capacity")
    return BufferConfig(
        name="uniform",
        capacity=capacity,
        batch_size=batch_size,
        min_size=min_size,
        updates_per_step=updates_per_step,
        save_dataset_path=str(config.get("save_dataset_path", "")),
        load_dataset_path=str(config.get("load_dataset_path", "")),
        offline_only=bool(config.get("offline_only", False)),
        offline_updates=int(config.get("offline_updates", 0)),
    )


def resolve_buffer_paths(config: BufferConfig, run_dir: Path) -> BufferConfig:
    save_path = _resolve_save_path(config.save_dataset_path, run_dir)
    load_path = _resolve_load_path(config.load_dataset_path, run_dir)
    return replace(config, save_dataset_path=save_path, load_dataset_path=load_path)


def initial_replay_buffer(config: BufferConfig) -> ReplayBufferState:
    if config.load_dataset_path:
        return load_replay_dataset(Path(config.load_dataset_path), capacity=config.capacity)
    capacity = max(config.capacity, 1)
    return ReplayBufferState(
        observations=jnp.zeros((capacity,), dtype=jnp.int32),
        actions=jnp.zeros((capacity,), dtype=jnp.int32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_observations=jnp.zeros((capacity,), dtype=jnp.int32),
        terminals=jnp.zeros((capacity,), dtype=jnp.bool_),
        size=jnp.asarray(0, dtype=jnp.int32),
        index=jnp.asarray(0, dtype=jnp.int32),
    )


def push_transition(
    state: ReplayBufferState,
    observation: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_observation: jax.Array,
    terminal: jax.Array,
) -> ReplayBufferState:
    capacity = state.observations.shape[0]
    index = state.index
    return ReplayBufferState(
        observations=state.observations.at[index].set(observation.astype(jnp.int32)),
        actions=state.actions.at[index].set(action.astype(jnp.int32)),
        rewards=state.rewards.at[index].set(reward.astype(jnp.float32)),
        next_observations=state.next_observations.at[index].set(next_observation.astype(jnp.int32)),
        terminals=state.terminals.at[index].set(terminal.astype(jnp.bool_)),
        size=jnp.minimum(state.size + 1, capacity).astype(jnp.int32),
        index=((index + 1) % capacity).astype(jnp.int32),
    )


def sample_batch(state: ReplayBufferState, key: jax.Array, batch_size: int) -> TransitionBatch:
    indices = jax.random.randint(key, (batch_size,), 0, state.size, dtype=jnp.int32)
    return TransitionBatch(
        observations=state.observations[indices],
        actions=state.actions[indices],
        rewards=state.rewards[indices],
        next_observations=state.next_observations[indices],
        terminals=state.terminals[indices],
    )


def save_replay_dataset(state: ReplayBufferState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = replay_dataset_arrays(state)
    np.savez_compressed(path, **dataset)


def load_replay_dataset(path: Path, *, capacity: int = 1) -> ReplayBufferState:
    if not path.exists():
        raise FileNotFoundError(f"Replay dataset does not exist: {path}")
    data = np.load(path)
    required = {"observations", "actions", "rewards", "next_observations", "terminals"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Replay dataset is missing arrays: {missing}")
    observations = np.asarray(data["observations"], dtype=np.int32).reshape(-1)
    actions = np.asarray(data["actions"], dtype=np.int32).reshape(-1)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    next_observations = np.asarray(data["next_observations"], dtype=np.int32).reshape(-1)
    terminals = np.asarray(data["terminals"], dtype=np.bool_).reshape(-1)
    size = len(observations)
    lengths = {len(actions), len(rewards), len(next_observations), len(terminals), size}
    if len(lengths) != 1:
        raise ValueError("Replay dataset arrays must all have the same length")
    if size == 0:
        raise ValueError("Replay dataset is empty")
    buffer_capacity = max(int(capacity), size, 1)
    padded = {
        "observations": np.zeros((buffer_capacity,), dtype=np.int32),
        "actions": np.zeros((buffer_capacity,), dtype=np.int32),
        "rewards": np.zeros((buffer_capacity,), dtype=np.float32),
        "next_observations": np.zeros((buffer_capacity,), dtype=np.int32),
        "terminals": np.zeros((buffer_capacity,), dtype=np.bool_),
    }
    padded["observations"][:size] = observations
    padded["actions"][:size] = actions
    padded["rewards"][:size] = rewards
    padded["next_observations"][:size] = next_observations
    padded["terminals"][:size] = terminals
    return ReplayBufferState(
        observations=jnp.asarray(padded["observations"]),
        actions=jnp.asarray(padded["actions"]),
        rewards=jnp.asarray(padded["rewards"]),
        next_observations=jnp.asarray(padded["next_observations"]),
        terminals=jnp.asarray(padded["terminals"]),
        size=jnp.asarray(size, dtype=jnp.int32),
        index=jnp.asarray(size % buffer_capacity, dtype=jnp.int32),
    )


def replay_dataset_arrays(state: ReplayBufferState) -> dict[str, np.ndarray]:
    size = int(np.asarray(jax.device_get(state.size)))
    return {
        "observations": np.asarray(jax.device_get(state.observations))[:size],
        "actions": np.asarray(jax.device_get(state.actions))[:size],
        "rewards": np.asarray(jax.device_get(state.rewards))[:size],
        "next_observations": np.asarray(jax.device_get(state.next_observations))[:size],
        "terminals": np.asarray(jax.device_get(state.terminals))[:size],
    }


def _resolve_save_path(path: str, run_dir: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    if candidate.is_absolute():
        return str(candidate)
    return str((run_dir / candidate).resolve())


def _resolve_load_path(path: str, run_dir: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists() or candidate.suffix != "":
            return str(candidate)
        suffixed = candidate.with_suffix(".npz")
        if suffixed.exists():
            return str(suffixed)
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    if candidate.suffix == "":
        suffixed = candidate.with_suffix(".npz")
        if suffixed.exists():
            return str(suffixed.resolve())
    run_candidate = (run_dir / candidate).resolve()
    if run_candidate.exists() or run_candidate.suffix != "":
        return str(run_candidate)
    run_suffixed = run_candidate.with_suffix(".npz")
    if run_suffixed.exists():
        return str(run_suffixed)
    return str(run_candidate)
