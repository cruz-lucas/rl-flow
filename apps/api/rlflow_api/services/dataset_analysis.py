from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ACTION_LABELS = ("Up", "Down", "Left", "Right")


@dataclass(frozen=True)
class GridDecoding:
    height: int
    width: int
    positions: np.ndarray
    valid_mask: np.ndarray
    source: str


@dataclass(frozen=True)
class VisitationAnalysis:
    height: int
    width: int
    action_count: int
    action_labels: tuple[str, ...]
    valid_mask: list[list[bool]]
    state_counts: list[list[int]]
    state_action_counts: list[list[list[int]]]
    source: str


@dataclass(frozen=True)
class OfflineRndAnalysis:
    algorithm: str
    granularity: str
    epochs: int
    batch_size: int
    unique_items: int
    loss_history: list[float]
    visitation: VisitationAnalysis | None
    learned_state_bonus: list[list[float | None]] | None
    count_state_bonus: list[list[float | None]] | None
    learned_state_action_bonus: list[list[list[float | None]]] | None
    count_state_action_bonus: list[list[list[float | None]]] | None
    scatter: list[dict[str, float | int | None]]


def transition_visitation(
    observations: np.ndarray,
    actions: np.ndarray,
) -> VisitationAnalysis | None:
    decoding = decode_grid_positions(observations)
    if decoding is None:
        return None
    action_values = np.asarray(actions, dtype=np.int32).reshape(-1)
    if action_values.size != decoding.positions.shape[0]:
        return None
    action_count = max(int(np.max(action_values, initial=0)) + 1, len(ACTION_LABELS))
    state_counts = np.zeros((decoding.height, decoding.width), dtype=np.int32)
    state_action_counts = np.zeros(
        (decoding.height, decoding.width, action_count),
        dtype=np.int32,
    )
    for (row, col), action in zip(decoding.positions, action_values, strict=True):
        if row < 0 or col < 0 or row >= decoding.height or col >= decoding.width:
            continue
        state_counts[row, col] += 1
        if 0 <= action < action_count:
            state_action_counts[row, col, action] += 1

    return VisitationAnalysis(
        height=decoding.height,
        width=decoding.width,
        action_count=action_count,
        action_labels=tuple(
            ACTION_LABELS[idx] if idx < len(ACTION_LABELS) else f"Action {idx}"
            for idx in range(action_count)
        ),
        valid_mask=decoding.valid_mask.astype(bool).tolist(),
        state_counts=state_counts.astype(int).tolist(),
        state_action_counts=state_action_counts.astype(int).tolist(),
        source=decoding.source,
    )


def offline_rnd_analysis(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    algorithm: Literal["rnd", "cfn", "classifier"] = "rnd",
    granularity: Literal["state", "state_action"],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_units: tuple[int, ...],
    activation: str,
    optimizer: str,
    action_conditioning: str,
    update_period: int,
    output_dim: int,
    intrinsic_reward_scale: float,
    intrinsic_stats_decay: float,
    intrinsic_reward_epsilon: float,
    intrinsic_reward_clip: float | None,
    intrinsic_reward_center: bool,
    max_grad_norm: float,
    seed: int,
    cfn_targets: np.ndarray | None = None,
) -> OfflineRndAnalysis:
    observation_features = _flatten_observations(observations)
    action_values = np.asarray(actions, dtype=np.int32).reshape(-1)
    if observation_features.shape[0] != action_values.shape[0]:
        raise ValueError("Observation and action arrays must have the same length")
    if observation_features.shape[0] == 0:
        raise ValueError("Dataset is empty")

    action_count = max(int(np.max(action_values, initial=0)) + 1, len(ACTION_LABELS))
    if granularity == "state_action":
        unique_keys, unique_indices, unique_counts = _unique_rows_with_counts(
            np.concatenate((observation_features, action_values[:, None]), axis=1)
        )
        del unique_keys
        unique_actions = action_values[unique_indices]
        eval_observations = observation_features[unique_indices]
        eval_actions = unique_actions
        display_actions = unique_actions
    else:
        unique_observations, unique_indices, unique_counts = _unique_rows_with_counts(
            observation_features
        )
        del unique_observations
        eval_observations = observation_features[unique_indices]
        display_actions = np.full(unique_indices.shape, -1, dtype=np.int32)
        if _is_no_action_conditioning(action_conditioning):
            eval_actions = np.zeros(unique_indices.shape, dtype=np.int32)
        else:
            eval_observations = np.repeat(eval_observations, action_count, axis=0)
            eval_actions = np.tile(
                np.arange(action_count, dtype=np.int32),
                unique_indices.shape[0],
            )

    if algorithm == "rnd":
        learned_bonus, loss_history = _train_rnd(
            observation_features,
            action_values,
            eval_observations,
            eval_actions,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            activation=activation,
            optimizer=optimizer,
            action_conditioning=action_conditioning,
            update_period=update_period,
            output_dim=output_dim,
            intrinsic_reward_scale=intrinsic_reward_scale,
            intrinsic_stats_decay=intrinsic_stats_decay,
            intrinsic_reward_epsilon=intrinsic_reward_epsilon,
            intrinsic_reward_clip=intrinsic_reward_clip,
            intrinsic_reward_center=intrinsic_reward_center,
            max_grad_norm=max_grad_norm,
            num_actions=action_count,
            seed=seed,
        )
    elif algorithm == "cfn":
        learned_bonus, loss_history = _train_cfn(
            observation_features,
            action_values,
            eval_observations,
            eval_actions,
            cfn_targets=cfn_targets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            activation=activation,
            optimizer=optimizer,
            action_conditioning=action_conditioning,
            update_period=update_period,
            output_dim=output_dim,
            intrinsic_reward_scale=intrinsic_reward_scale,
            intrinsic_stats_decay=intrinsic_stats_decay,
            intrinsic_reward_epsilon=intrinsic_reward_epsilon,
            intrinsic_reward_clip=intrinsic_reward_clip,
            intrinsic_reward_center=intrinsic_reward_center,
            max_grad_norm=max_grad_norm,
            num_actions=action_count,
            seed=seed,
        )
    elif algorithm == "classifier":
        learned_bonus, loss_history = _train_known_unknown_classifier(
            observation_features,
            action_values,
            eval_observations,
            eval_actions,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            activation=activation,
            optimizer=optimizer,
            action_conditioning=action_conditioning,
            max_grad_norm=max_grad_norm,
            num_actions=action_count,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported offline RL algorithm: {algorithm}")
    if granularity == "state" and not _is_no_action_conditioning(action_conditioning):
        learned_bonus = learned_bonus.reshape(unique_indices.shape[0], action_count).mean(
            axis=1
        )
    count_bonus = 1.0 / (np.sqrt(unique_counts.astype(np.float32)) + 1e-8)
    visitation = transition_visitation(observations, actions)
    decoding = decode_grid_positions(observations)

    learned_state_bonus = None
    count_state_bonus = None
    learned_state_action_bonus = None
    count_state_action_bonus = None
    scatter: list[dict[str, float | int | None]] = []

    if decoding is not None:
        unique_positions = decoding.positions[unique_indices]
        if granularity == "state_action":
            learned_state_action_bonus = _empty_3d(
                decoding.height,
                decoding.width,
                action_count,
            )
            count_state_action_bonus = _empty_3d(
                decoding.height,
                decoding.width,
                action_count,
            )
            for idx, ((row, col), action) in enumerate(
                zip(unique_positions, display_actions, strict=True)
            ):
                row_i = int(row)
                col_i = int(col)
                action_i = int(action)
                if (
                    row_i < 0
                    or col_i < 0
                    or row_i >= decoding.height
                    or col_i >= decoding.width
                    or action_i < 0
                    or action_i >= action_count
                ):
                    continue
                learned_state_action_bonus[row_i][col_i][action_i] = float(
                    learned_bonus[idx]
                )
                count_state_action_bonus[row_i][col_i][action_i] = float(
                    count_bonus[idx]
                )
                scatter.append(
                    _scatter_point(
                        unique_counts[idx],
                        learned_bonus[idx],
                        count_bonus[idx],
                        row=row_i,
                        col=col_i,
                        action=action_i,
                    )
                )
        else:
            learned_state_bonus = _weighted_grid(
                decoding.height,
                decoding.width,
                unique_positions,
                learned_bonus,
                unique_counts,
            )
            count_state_bonus = _weighted_grid(
                decoding.height,
                decoding.width,
                unique_positions,
                count_bonus,
                unique_counts,
            )
            for idx, (row, col) in enumerate(unique_positions):
                scatter.append(
                    _scatter_point(
                        unique_counts[idx],
                        learned_bonus[idx],
                        count_bonus[idx],
                        row=int(row),
                        col=int(col),
                        action=None,
                    )
                )
    else:
        for idx in range(unique_counts.shape[0]):
            scatter.append(
                _scatter_point(
                    unique_counts[idx],
                    learned_bonus[idx],
                    count_bonus[idx],
                    row=None,
                    col=None,
                    action=(int(display_actions[idx]) if display_actions[idx] >= 0 else None),
                )
            )

    return OfflineRndAnalysis(
        algorithm=algorithm,
        granularity=granularity,
        epochs=epochs,
        batch_size=batch_size,
        unique_items=int(unique_counts.shape[0]),
        loss_history=loss_history,
        visitation=visitation,
        learned_state_bonus=learned_state_bonus,
        count_state_bonus=count_state_bonus,
        learned_state_action_bonus=learned_state_action_bonus,
        count_state_action_bonus=count_state_action_bonus,
        scatter=scatter,
    )


def _is_no_action_conditioning(mode: str) -> bool:
    return str(mode).strip().lower() in {"none", "state", "observation", "false"}


def decode_grid_positions(observations: np.ndarray) -> GridDecoding | None:
    values = np.asarray(observations)
    if values.ndim == 1:
        return _decode_scalar_grid(values)
    if values.ndim == 2:
        symbolic = _decode_symbolic_flat(values)
        if symbolic is not None:
            return symbolic
        return _decode_feature_grid(values)
    if values.ndim == 4 and values.shape[-1] == 3:
        return _decode_symbolic_images(values)
    return None


def _decode_scalar_grid(values: np.ndarray) -> GridDecoding | None:
    states = np.asarray(values, dtype=np.int32).reshape(-1)
    if states.size == 0:
        return None
    num_states = int(np.max(states)) + 1
    direction_factor = 1
    inner_side = int(round(np.sqrt(num_states)))
    if inner_side * inner_side != num_states and num_states % 4 == 0:
        direction_factor = 4
        inner_side = int(round(np.sqrt(num_states // direction_factor)))
    if inner_side * inner_side * direction_factor != num_states:
        return None
    position_index = states // direction_factor
    positions = np.stack(
        (
            position_index // inner_side + 1,
            position_index % inner_side + 1,
        ),
        axis=1,
    ).astype(np.int32)
    valid_mask = np.zeros((inner_side + 2, inner_side + 2), dtype=bool)
    valid_mask[1:-1, 1:-1] = True
    return GridDecoding(
        height=inner_side + 2,
        width=inner_side + 2,
        positions=positions,
        valid_mask=valid_mask,
        source="scalar_tabular",
    )


def _decode_symbolic_flat(values: np.ndarray) -> GridDecoding | None:
    feature_dim = values.shape[1]
    if feature_dim % 3 != 0:
        return None
    side = int(round(np.sqrt(feature_dim // 3)))
    if side * side * 3 != feature_dim:
        return None
    return _decode_symbolic_images(values.reshape(values.shape[0], side, side, 3))


def _decode_symbolic_images(values: np.ndarray) -> GridDecoding | None:
    raw = np.asarray(values)
    if np.max(raw, initial=0.0) <= 1.0:
        raw = np.rint(raw * 255.0)
    else:
        raw = np.rint(raw)
    raw = raw.astype(np.int32)
    player_mask = raw[..., 0] == 10
    if not np.any(player_mask):
        return None
    positions = np.full((raw.shape[0], 2), -1, dtype=np.int32)
    for idx, sample_mask in enumerate(player_mask):
        coords = np.argwhere(sample_mask)
        if coords.size:
            positions[idx] = coords[0]
    first = raw[0]
    valid_mask = first[..., 0] != 2
    return GridDecoding(
        height=int(raw.shape[1]),
        width=int(raw.shape[2]),
        positions=positions,
        valid_mask=valid_mask,
        source="navix_symbolic",
    )


def _decode_feature_grid(values: np.ndarray) -> GridDecoding | None:
    feature_dim = values.shape[1]
    inner_side = int(round(np.sqrt(feature_dim)))
    direction_features = 0
    if inner_side * inner_side != feature_dim:
        maybe_inner_side = int(round(np.sqrt(max(feature_dim - 4, 0))))
        if maybe_inner_side * maybe_inner_side + 4 != feature_dim:
            return None
        inner_side = maybe_inner_side
        direction_features = 4
    position_features = values[:, : feature_dim - direction_features]
    position_index = np.argmax(position_features, axis=1).astype(np.int32)
    positions = np.stack(
        (
            position_index // inner_side + 1,
            position_index % inner_side + 1,
        ),
        axis=1,
    ).astype(np.int32)
    valid_mask = np.zeros((inner_side + 2, inner_side + 2), dtype=bool)
    valid_mask[1:-1, 1:-1] = True
    return GridDecoding(
        height=inner_side + 2,
        width=inner_side + 2,
        positions=positions,
        valid_mask=valid_mask,
        source="navix_feature",
    )


def _flatten_observations(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    return values.reshape((values.shape[0], -1))


def _one_hot(values: np.ndarray, count: int) -> np.ndarray:
    encoded = np.zeros((values.shape[0], count), dtype=np.float32)
    encoded[np.arange(values.shape[0]), values.astype(np.int32)] = 1.0
    return encoded


def _unique_rows_with_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_values, unique_indices, unique_counts = np.unique(
        values,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    order = np.argsort(unique_indices)
    return (
        unique_values[order],
        unique_indices[order].astype(np.int32),
        unique_counts[order].astype(np.int32),
    )


def _train_rnd(
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    eval_observations: np.ndarray,
    eval_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_units: tuple[int, ...],
    activation: str,
    optimizer: str,
    action_conditioning: str,
    update_period: int,
    output_dim: int,
    intrinsic_reward_scale: float,
    intrinsic_stats_decay: float,
    intrinsic_reward_epsilon: float,
    intrinsic_reward_clip: float | None,
    intrinsic_reward_center: bool,
    max_grad_norm: float,
    num_actions: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    import jax
    import jax.numpy as jnp

    from rlflow_builtin.dqn.training import (
        DqnAgentConfig,
        DqnIntrinsicConfig,
        _canonicalize_action_conditioning,
        _initial_intrinsic_state,
        _normalize_intrinsic_reward,
        _optimizer,
        _rnd_prediction_error,
        _rnd_update,
    )

    train_observations_array = jnp.asarray(train_observations, dtype=jnp.float32)
    train_actions_array = jnp.asarray(train_actions, dtype=jnp.int32)
    eval_observations_array = jnp.asarray(eval_observations, dtype=jnp.float32)
    eval_actions_array = jnp.asarray(eval_actions, dtype=jnp.int32)
    input_dim = int(train_observations.shape[1])
    key = jax.random.PRNGKey(seed)
    key, intrinsic_key = jax.random.split(key)
    agent = DqnAgentConfig(
        algorithm="dqn",
        learning_rate=learning_rate,
        discount=0.99,
        hidden_units=hidden_units,
        activation=activation,
        update_frequency=1,
        target_update_frequency=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        eval_epsilon=0.0,
        loss_type="mse",
        huber_delta=1.0,
        double_q=False,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
        optimizer_epsilon=1e-8,
        optimizer_weight_decay=0.0,
        optimizer_momentum=0.0,
        optimizer_decay=0.95,
        optimizer_centered=False,
        normalize_observations=False,
        obs_normalization_epsilon=1e-8,
        obs_normalization_clip=5.0,
        rmax_bonus_threshold=0.5,
        rmax_v_max=100.0,
        seed=seed,
    )
    intrinsic = DqnIntrinsicConfig(
        kind="rnd",
        intrinsic_reward_scale=intrinsic_reward_scale,
        intrinsic_stats_decay=intrinsic_stats_decay,
        intrinsic_reward_epsilon=intrinsic_reward_epsilon,
        intrinsic_reward_clip=intrinsic_reward_clip,
        intrinsic_reward_center=intrinsic_reward_center,
        hidden_units=hidden_units,
        activation=activation,
        output_dim=output_dim,
        optimizer=optimizer,
        learning_rate=learning_rate,
        action_conditioning=_canonicalize_action_conditioning(action_conditioning),
        update_period=update_period,
    )
    intrinsic_state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim,
        num_actions,
        intrinsic_key,
    )
    intrinsic_optimizer = _optimizer(agent, intrinsic.learning_rate, intrinsic.optimizer)

    @jax.jit
    def update(state, batch, next_gradient_step):
        _bonus, state, loss = _rnd_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
        return state, loss

    num_rows = int(train_observations.shape[0])
    batch_size = max(1, min(int(batch_size), num_rows))
    loss_history: list[float] = []
    gradient_step = 0
    for _epoch in range(int(epochs)):
        key, epoch_key = jax.random.split(key)
        indices = np.asarray(jax.random.permutation(epoch_key, num_rows))
        epoch_losses: list[float] = []
        for start in range(0, num_rows, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = {
                "observations": train_observations_array[batch_indices],
                "actions": train_actions_array[batch_indices],
                "rewards": jnp.zeros((batch_indices.shape[0],), dtype=jnp.float32),
                "next_observations": train_observations_array[batch_indices],
                "terminals": jnp.zeros((batch_indices.shape[0],), dtype=jnp.float32),
            }
            gradient_step += 1
            intrinsic_state, loss = update(
                intrinsic_state,
                batch,
                jnp.asarray(gradient_step, dtype=jnp.int32),
            )
            epoch_losses.append(float(loss))
        loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    @jax.jit
    def evaluate(state, observations, actions):
        raw_bonus, _intrinsic_input, _target_features = _rnd_prediction_error(
            state.target_params,
            state.predictor_params,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
        normalized_bonus = _normalize_intrinsic_reward(
            intrinsic,
            raw_bonus,
            state.reward_mean,
            state.reward_var,
        )
        return intrinsic.intrinsic_reward_scale * normalized_bonus

    learned_bonus = np.asarray(
        evaluate(intrinsic_state, eval_observations_array, eval_actions_array)
    )
    return learned_bonus.astype(np.float32), loss_history


def _train_cfn(
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    eval_observations: np.ndarray,
    eval_actions: np.ndarray,
    *,
    cfn_targets: np.ndarray | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_units: tuple[int, ...],
    activation: str,
    optimizer: str,
    action_conditioning: str,
    update_period: int,
    output_dim: int,
    intrinsic_reward_scale: float,
    intrinsic_stats_decay: float,
    intrinsic_reward_epsilon: float,
    intrinsic_reward_clip: float | None,
    intrinsic_reward_center: bool,
    max_grad_norm: float,
    num_actions: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    import jax
    import jax.numpy as jnp

    from rlflow_builtin.dqn.training import (
        DqnIntrinsicConfig,
        _canonicalize_action_conditioning,
        _cfn_outputs,
        _cfn_update,
        _initial_intrinsic_state,
        _normalize_intrinsic_reward,
        _optimizer,
    )

    train_observations_array = jnp.asarray(train_observations, dtype=jnp.float32)
    train_actions_array = jnp.asarray(train_actions, dtype=jnp.int32)
    eval_observations_array = jnp.asarray(eval_observations, dtype=jnp.float32)
    eval_actions_array = jnp.asarray(eval_actions, dtype=jnp.int32)
    input_dim = int(train_observations.shape[1])
    key = jax.random.PRNGKey(seed)
    key, intrinsic_key, target_key = jax.random.split(key, 3)
    agent = _offline_agent_config(
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        activation=activation,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        seed=seed,
    )
    intrinsic = DqnIntrinsicConfig(
        kind="cfn",
        intrinsic_reward_scale=intrinsic_reward_scale,
        intrinsic_stats_decay=intrinsic_stats_decay,
        intrinsic_reward_epsilon=intrinsic_reward_epsilon,
        intrinsic_reward_clip=intrinsic_reward_clip,
        intrinsic_reward_center=intrinsic_reward_center,
        hidden_units=hidden_units,
        activation=activation,
        output_dim=output_dim,
        optimizer=optimizer,
        learning_rate=learning_rate,
        action_conditioning=_canonicalize_action_conditioning(action_conditioning),
        update_period=update_period,
        cfn_use_random_prior=True,
        cfn_prior_scale=1.0,
        cfn_bonus_exponent=0.5,
        cfn_final_tanh=False,
    )
    intrinsic_state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim,
        num_actions,
        intrinsic_key,
    )
    intrinsic_optimizer = _optimizer(agent, intrinsic.learning_rate, intrinsic.optimizer)
    train_targets = _cfn_training_targets(
        cfn_targets,
        train_observations.shape[0],
        output_dim,
        target_key,
    )

    @jax.jit
    def update(state, batch, next_gradient_step):
        _bonus, state, loss = _cfn_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
        return state, loss

    num_rows = int(train_observations.shape[0])
    batch_size = max(1, min(int(batch_size), num_rows))
    loss_history: list[float] = []
    gradient_step = 0
    for _epoch in range(int(epochs)):
        key, epoch_key = jax.random.split(key)
        indices = np.asarray(jax.random.permutation(epoch_key, num_rows))
        epoch_losses: list[float] = []
        for start in range(0, num_rows, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = {
                "observations": train_observations_array[batch_indices],
                "actions": train_actions_array[batch_indices],
                "rewards": jnp.zeros((batch_indices.shape[0],), dtype=jnp.float32),
                "next_observations": train_observations_array[batch_indices],
                "terminals": jnp.zeros((batch_indices.shape[0],), dtype=jnp.float32),
                "intrinsic_targets": train_targets[batch_indices],
            }
            gradient_step += 1
            intrinsic_state, loss = update(
                intrinsic_state,
                batch,
                jnp.asarray(gradient_step, dtype=jnp.int32),
            )
            epoch_losses.append(float(loss))
        loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    @jax.jit
    def evaluate(state, observations, actions):
        raw_bonus, _intrinsic_input, _prior_features, _predictor_features, _coin_flips = _cfn_outputs(
            state.prior_params,
            state.predictor_params,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
        normalized_bonus = _normalize_intrinsic_reward(
            intrinsic,
            raw_bonus,
            state.reward_mean,
            state.reward_var,
        )
        return intrinsic.intrinsic_reward_scale * normalized_bonus

    learned_bonus = np.asarray(
        evaluate(intrinsic_state, eval_observations_array, eval_actions_array)
    )
    return learned_bonus.astype(np.float32), loss_history


def _train_known_unknown_classifier(
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    eval_observations: np.ndarray,
    eval_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_units: tuple[int, ...],
    activation: str,
    optimizer: str,
    action_conditioning: str,
    max_grad_norm: float,
    num_actions: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    import jax
    import jax.numpy as jnp
    import optax

    from rlflow_builtin.dqn.training import (
        _apply_mlp,
        _canonicalize_action_conditioning,
        _conditioned_input,
        _conditioned_input_dim,
        _conditioned_output_dim,
        _init_mlp,
        _optimizer,
        _select_conditioned_features,
    )

    mode = _canonicalize_action_conditioning(action_conditioning)
    rng = np.random.default_rng(seed + 97)
    negative_observations = _negative_observation_samples(train_observations, rng)
    negative_actions = rng.integers(0, num_actions, size=train_actions.shape[0], dtype=np.int32)
    classifier_observations = np.concatenate(
        (train_observations, negative_observations),
        axis=0,
    ).astype(np.float32)
    classifier_actions = np.concatenate((train_actions, negative_actions), axis=0).astype(
        np.int32
    )
    labels = np.concatenate(
        (
            np.zeros((train_observations.shape[0],), dtype=np.float32),
            np.ones((negative_observations.shape[0],), dtype=np.float32),
        ),
        axis=0,
    )

    observations_array = jnp.asarray(classifier_observations, dtype=jnp.float32)
    actions_array = jnp.asarray(classifier_actions, dtype=jnp.int32)
    labels_array = jnp.asarray(labels, dtype=jnp.float32)
    eval_observations_array = jnp.asarray(eval_observations, dtype=jnp.float32)
    eval_actions_array = jnp.asarray(eval_actions, dtype=jnp.int32)
    agent = _offline_agent_config(
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        activation=activation,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        seed=seed,
    )
    input_dim = _conditioned_input_dim(int(train_observations.shape[1]), num_actions, mode)
    output_dim = _conditioned_output_dim(num_actions, 1, mode)
    key = jax.random.PRNGKey(seed)
    key, params_key = jax.random.split(key)
    params = _init_mlp(params_key, input_dim, hidden_units, output_dim)
    classifier_optimizer = _optimizer(agent, learning_rate, optimizer)
    opt_state = classifier_optimizer.init(params)

    def logits(params, observations, actions):
        conditioned = _conditioned_input(observations, actions, mode, num_actions)
        features = _apply_mlp(params, conditioned, activation)
        return _select_conditioned_features(
            features,
            actions,
            mode,
            1,
            num_actions,
        ).squeeze(-1)

    @jax.jit
    def update(params, opt_state, observations, actions, labels):
        def loss_fn(current_params):
            batch_logits = logits(current_params, observations, actions)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(batch_logits, labels))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = classifier_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    num_rows = int(classifier_observations.shape[0])
    batch_size = max(1, min(int(batch_size), num_rows))
    loss_history: list[float] = []
    for _epoch in range(int(epochs)):
        key, epoch_key = jax.random.split(key)
        indices = np.asarray(jax.random.permutation(epoch_key, num_rows))
        epoch_losses: list[float] = []
        for start in range(0, num_rows, batch_size):
            batch_indices = indices[start : start + batch_size]
            params, opt_state, loss = update(
                params,
                opt_state,
                observations_array[batch_indices],
                actions_array[batch_indices],
                labels_array[batch_indices],
            )
            epoch_losses.append(float(loss))
        loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    learned_bonus = np.asarray(jax.nn.sigmoid(logits(params, eval_observations_array, eval_actions_array)))
    return learned_bonus.astype(np.float32), loss_history


def _offline_agent_config(
    *,
    learning_rate: float,
    hidden_units: tuple[int, ...],
    activation: str,
    optimizer: str,
    max_grad_norm: float,
    seed: int,
):
    from rlflow_builtin.dqn.training import DqnAgentConfig

    return DqnAgentConfig(
        algorithm="dqn",
        learning_rate=learning_rate,
        discount=0.99,
        hidden_units=hidden_units,
        activation=activation,
        update_frequency=1,
        target_update_frequency=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        eval_epsilon=0.0,
        loss_type="mse",
        huber_delta=1.0,
        double_q=False,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
        optimizer_epsilon=1e-8,
        optimizer_weight_decay=0.0,
        optimizer_momentum=0.0,
        optimizer_decay=0.95,
        optimizer_centered=False,
        normalize_observations=False,
        obs_normalization_epsilon=1e-8,
        obs_normalization_clip=5.0,
        rmax_bonus_threshold=0.5,
        rmax_v_max=100.0,
        seed=seed,
    )


def _cfn_training_targets(
    cfn_targets: np.ndarray | None,
    num_rows: int,
    output_dim: int,
    key,
):
    import jax
    import jax.numpy as jnp

    if cfn_targets is not None:
        targets = np.asarray(cfn_targets, dtype=np.float32)
        if targets.ndim == 2 and targets.shape[0] == num_rows and targets.shape[1] == output_dim:
            return jnp.asarray(targets, dtype=jnp.float32)
    sampled = jax.random.bernoulli(key, p=0.5, shape=(num_rows, output_dim))
    return jnp.where(sampled, 1.0, -1.0).astype(jnp.float32)


def _negative_observation_samples(observations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    num_rows, feature_dim = values.shape
    if feature_dim > 1 and _looks_one_hot(values):
        indices = rng.integers(0, feature_dim, size=num_rows)
        output = np.zeros_like(values)
        output[np.arange(num_rows), indices] = 1.0
        return output

    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    if feature_dim == 1 and np.allclose(values, np.rint(values)):
        low = int(np.floor(float(minimum[0])))
        high = int(np.ceil(float(maximum[0]))) + 1
        if high <= low:
            high = low + 1
        return rng.integers(low, high, size=(num_rows, 1)).astype(np.float32)

    span = maximum - minimum
    sampled = rng.uniform(minimum, maximum, size=values.shape).astype(np.float32)
    fixed_columns = span <= 1e-8
    if np.any(fixed_columns):
        sampled[:, fixed_columns] = values[:, fixed_columns]
    return sampled


def _looks_one_hot(values: np.ndarray) -> bool:
    if values.ndim != 2 or values.shape[1] <= 1:
        return False
    row_sums = np.sum(values, axis=1)
    return bool(
        np.allclose(row_sums, 1.0, atol=1e-5)
        and np.all((np.isclose(values, 0.0, atol=1e-5)) | (np.isclose(values, 1.0, atol=1e-5)))
    )


def _empty_3d(height: int, width: int, depth: int) -> list[list[list[float | None]]]:
    return [[[None for _ in range(depth)] for _ in range(width)] for _ in range(height)]


def _weighted_grid(
    height: int,
    width: int,
    positions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> list[list[float | None]]:
    value_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    for (row, col), value, weight in zip(positions, values, weights, strict=True):
        row_i = int(row)
        col_i = int(col)
        if row_i < 0 or col_i < 0 or row_i >= height or col_i >= width:
            continue
        value_sum[row_i, col_i] += float(value) * float(weight)
        weight_sum[row_i, col_i] += float(weight)
    output: list[list[float | None]] = []
    for row in range(height):
        output_row: list[float | None] = []
        for col in range(width):
            if weight_sum[row, col] <= 0.0:
                output_row.append(None)
            else:
                output_row.append(float(value_sum[row, col] / weight_sum[row, col]))
        output.append(output_row)
    return output


def _scatter_point(
    count: int | np.integer,
    learned_bonus: float | np.floating,
    count_bonus: float | np.floating,
    *,
    row: int | None,
    col: int | None,
    action: int | None,
) -> dict[str, float | int | None]:
    return {
        "count": int(count),
        "learned_bonus": float(learned_bonus),
        "count_bonus": float(count_bonus),
        "row": row,
        "col": col,
        "action": action,
    }
