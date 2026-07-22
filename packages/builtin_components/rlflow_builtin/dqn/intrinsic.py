"""DQN intrinsic-reward subsystem: RND, CFN, count-based, and SimHash modules.

Extracted from ``dqn.training``: state init, per-kind updates, bonus computation,
normalization statistics, count/simhash key tables, and the role-based dispatch
used for the knownness-signal and reward-intrinsic ports. Re-exported by
``dqn.training`` for backwards compatibility.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlflow_builtin.dqn.config import (
    ActionConditioning,
    DqnAgentConfig,
    DqnIntrinsicConfig,
)
from rlflow_builtin.dqn.networks import (
    _apply_autoencoder_encoder,
    _apply_mlp,
    _init_autoencoder,
    _init_mlp,
    _optimizer,
)
from rlflow_builtin.dqn.state import DqnIntrinsicState, DqnTrainState


def _intrinsic_update_roles_from_batch(
    state: DqnTrainState,
    batch: dict[str, jax.Array],
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    shared_intrinsic: bool,
    intrinsic_optimizer: optax.GradientTransformation,
    reward_intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
) -> tuple[DqnTrainState, jax.Array]:
    if shared_intrinsic:
        intrinsic_state, intrinsic_step, intrinsic_loss = _intrinsic_update_role(
            state.intrinsic_state,
            state.intrinsic_gradient_step,
            batch,
            knownness,
            intrinsic_optimizer,
            num_actions,
            target_field="intrinsic_targets",
        )
        return (
            state._replace(
                intrinsic_state=intrinsic_state,
                reward_intrinsic_state=intrinsic_state,
                intrinsic_gradient_step=intrinsic_step,
                reward_intrinsic_gradient_step=intrinsic_step,
            ),
            intrinsic_loss,
        )

    intrinsic_state, intrinsic_step, intrinsic_loss = _intrinsic_update_role(
        state.intrinsic_state,
        state.intrinsic_gradient_step,
        batch,
        knownness,
        intrinsic_optimizer,
        num_actions,
        target_field="intrinsic_targets",
    )
    reward_intrinsic_state, reward_intrinsic_step, reward_intrinsic_loss = _intrinsic_update_role(
        state.reward_intrinsic_state,
        state.reward_intrinsic_gradient_step,
        batch,
        intrinsic_reward,
        reward_intrinsic_optimizer,
        num_actions,
        target_field="reward_intrinsic_targets",
    )
    return (
        state._replace(
            intrinsic_state=intrinsic_state,
            reward_intrinsic_state=reward_intrinsic_state,
            intrinsic_gradient_step=intrinsic_step,
            reward_intrinsic_gradient_step=reward_intrinsic_step,
        ),
        intrinsic_loss + reward_intrinsic_loss,
    )


def _intrinsic_update_role(
    intrinsic_state: DqnIntrinsicState,
    intrinsic_gradient_step: jax.Array,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    *,
    target_field: str,
) -> tuple[DqnIntrinsicState, jax.Array, jax.Array]:
    if intrinsic.kind == "none":
        return (
            intrinsic_state,
            intrinsic_gradient_step,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    next_gradient_step = intrinsic_gradient_step + 1
    role_batch = batch
    if target_field != "intrinsic_targets":
        role_batch = {**batch, "intrinsic_targets": batch[target_field]}
    _intrinsic_rewards, intrinsic_state, intrinsic_loss = _intrinsic_update(
        intrinsic_state,
        role_batch,
        intrinsic,
        intrinsic_optimizer,
        num_actions,
        next_gradient_step,
    )
    return intrinsic_state, next_gradient_step, intrinsic_loss


def _batch_intrinsic_rewards(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    if intrinsic.kind == "none":
        return jnp.zeros_like(batch["rewards"])
    if _count_uses_oracle_tabular(intrinsic):
        return _count_direct_bonus(
            state,
            batch["state_ids"],
            batch["actions"],
            intrinsic,
            num_actions,
        )
    return _intrinsic_bonus(
        state,
        batch["observations"],
        batch["actions"],
        intrinsic,
        num_actions,
    )


def _intrinsic_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    if intrinsic.kind == "none":
        return (
            jnp.zeros_like(batch["rewards"]),
            state,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    if intrinsic.kind == "rnd":
        return _rnd_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
    if intrinsic.kind == "cfn":
        return _cfn_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
    if intrinsic.kind == "simhash":
        return _simhash_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
    return _count_update(
        state,
        batch,
        intrinsic,
        num_actions,
    )


def _rnd_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    actions = batch["actions"]
    # TODO: add next state or current state option
    intrinsic_observations = (
        # batch["next_observations"]
        # if intrinsic.action_conditioning == "none"
        # else batch["observations"]
        batch["observations"]
    )
    prediction_error, intrinsic_input, target_features = _rnd_prediction_error(
        state.target_params,
        state.predictor_params,
        intrinsic_observations,
        actions,
        intrinsic,
        num_actions,
    )
    intrinsic_reward = _normalize_intrinsic_reward(
        intrinsic,
        prediction_error,
        state.reward_mean,
        state.reward_var,
    )

    def loss_fn(predictor_params):
        predictor_features = _select_conditioned_features(
            _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
        return jnp.mean(jnp.square(predictor_features - target_features))

    def do_predictor_update(args):
        predictor_params, opt_state = args
        intrinsic_loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = intrinsic_optimizer.update(
            grads,
            opt_state,
            predictor_params,
        )
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, intrinsic_loss

    def skip_predictor_update(args):
        predictor_params, opt_state = args
        return predictor_params, opt_state, loss_fn(predictor_params)

    predictor_params, opt_state, intrinsic_loss = jax.lax.cond(
        next_gradient_step % intrinsic.update_period == 0,
        do_predictor_update,
        skip_predictor_update,
        (state.predictor_params, state.opt_state),
    )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(prediction_error),
    )
    return (
        intrinsic_reward,
        state._replace(
            predictor_params=predictor_params,
            opt_state=opt_state,
            reward_mean=reward_mean,
            reward_var=reward_var,
        ),
        intrinsic_loss,
    )


def _cfn_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    actions = batch["actions"]
    # TODO: add next state or current state option
    intrinsic_observations = (
        # batch["next_observations"]
        # if intrinsic.action_conditioning == "none"
        # else batch["observations"]
        batch["observations"]
    )
    raw_bonus, intrinsic_input, prior_features, _predictor_features, _coin_flips = _cfn_outputs(
        state.prior_params,
        state.predictor_params,
        intrinsic_observations,
        actions,
        intrinsic,
        num_actions,
    )
    intrinsic_reward = _normalize_intrinsic_reward(
        intrinsic,
        raw_bonus,
        state.reward_mean,
        state.reward_var,
    )
    targets = batch["intrinsic_targets"]

    def loss_fn(predictor_params):
        predictor_features = _maybe_tanh(
            _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
            intrinsic.cfn_final_tanh,
        )
        predictor_features = _select_conditioned_features(
            predictor_features,
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
        if intrinsic.cfn_use_random_prior:
            coin_flips = predictor_features + intrinsic.cfn_prior_scale * prior_features
        else:
            coin_flips = predictor_features
        return jnp.mean(jnp.square(coin_flips - targets))

    def do_predictor_update(args):
        predictor_params, opt_state = args
        intrinsic_loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = intrinsic_optimizer.update(
            grads,
            opt_state,
            predictor_params,
        )
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, intrinsic_loss

    def skip_predictor_update(args):
        predictor_params, opt_state = args
        return predictor_params, opt_state, loss_fn(predictor_params)

    predictor_params, opt_state, intrinsic_loss = jax.lax.cond(
        next_gradient_step % intrinsic.update_period == 0,
        do_predictor_update,
        skip_predictor_update,
        (state.predictor_params, state.opt_state),
    )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(raw_bonus),
    )
    return (
        intrinsic_reward,
        state._replace(
            predictor_params=predictor_params,
            opt_state=opt_state,
            reward_mean=reward_mean,
            reward_var=reward_var,
        ),
        intrinsic_loss,
    )


def _count_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    raw_bonus = (
        _count_direct_bonus(
            state,
            batch["state_ids"],
            batch["actions"],
            intrinsic,
            num_actions,
        )
        if _count_uses_oracle_tabular(intrinsic)
        else _count_raw_bonus(
            state,
            batch["observations"],
            batch["actions"],
            intrinsic,
            num_actions,
        )
    )
    intrinsic_reward = raw_bonus
    # _normalize_intrinsic_reward(
    #     intrinsic,
    #     raw_bonus,
    #     state.reward_mean,
    #     state.reward_var,
    # )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(raw_bonus),
    )
    return (
        intrinsic_reward,
        state._replace(reward_mean=reward_mean, reward_var=reward_var),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _simhash_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    raw_bonus = _simhash_raw_bonus(
        state,
        batch["observations"],
        batch["actions"],
        intrinsic,
        num_actions,
    )
    predictor_params = state.predictor_params
    opt_state = state.opt_state
    intrinsic_loss = jnp.asarray(0.0, dtype=jnp.float32)

    if intrinsic.simhash_mode == "learned":
        intrinsic_input = _simhash_input(
            batch["observations"],
            batch["actions"],
            intrinsic,
            num_actions,
        )

        def loss_fn(autoencoder_params):
            reconstruction = _apply_mlp(autoencoder_params, intrinsic_input, intrinsic.activation)
            return jnp.mean(jnp.square(reconstruction - intrinsic_input))

        def do_autoencoder_update(args):
            autoencoder_params, current_opt_state = args
            loss, grads = jax.value_and_grad(loss_fn)(autoencoder_params)
            updates, current_opt_state = intrinsic_optimizer.update(
                grads,
                current_opt_state,
                autoencoder_params,
            )
            autoencoder_params = optax.apply_updates(autoencoder_params, updates)
            return autoencoder_params, current_opt_state, loss

        def skip_autoencoder_update(args):
            autoencoder_params, current_opt_state = args
            return autoencoder_params, current_opt_state, loss_fn(autoencoder_params)

        predictor_params, opt_state, intrinsic_loss = jax.lax.cond(
            next_gradient_step % intrinsic.update_period == 0,
            do_autoencoder_update,
            skip_autoencoder_update,
            (state.predictor_params, state.opt_state),
        )

    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(raw_bonus),
    )
    return (
        raw_bonus,
        state._replace(
            predictor_params=predictor_params,
            opt_state=opt_state,
            reward_mean=reward_mean,
            reward_var=reward_var,
        ),
        intrinsic_loss,
    )


def _initial_intrinsic_state(
    agent: DqnAgentConfig,
    intrinsic: DqnIntrinsicConfig,
    input_dim: int,
    num_actions: int,
    key: jax.Array,
    oracle_state_space_size: int | None = None,
) -> DqnIntrinsicState:
    key, target_key, prior_key, predictor_key = jax.random.split(key, 4)
    del key
    if intrinsic.kind == "none":
        target_input_dim = 1
        target_output_dim = 1
        hidden_units: tuple[int, ...] = ()
        target_params = _init_mlp(target_key, target_input_dim, hidden_units, target_output_dim)
        prior_params = _init_mlp(prior_key, target_input_dim, hidden_units, target_output_dim)
        predictor_params = _init_mlp(
            predictor_key,
            target_input_dim,
            hidden_units,
            target_output_dim,
        )
    elif intrinsic.kind == "simhash":
        target_input_dim = _conditioned_input_dim(
            input_dim,
            num_actions,
            intrinsic.action_conditioning,
        )
        projection_input_dim = (
            intrinsic.output_dim if intrinsic.simhash_mode == "learned" else target_input_dim
        )
        target_params = _init_mlp(target_key, projection_input_dim, (), intrinsic.simhash_bits)
        prior_params = _init_mlp(prior_key, 1, (), 1)
        if intrinsic.simhash_mode == "learned":
            predictor_params = _init_autoencoder(
                predictor_key,
                target_input_dim,
                intrinsic.hidden_units,
                intrinsic.output_dim,
            )
        else:
            predictor_params = _init_mlp(predictor_key, 1, (), 1)
    else:
        target_input_dim = _conditioned_input_dim(
            input_dim,
            num_actions,
            intrinsic.action_conditioning,
        )
        target_output_dim = _conditioned_output_dim(
            num_actions,
            intrinsic.output_dim,
            intrinsic.action_conditioning,
        )
        hidden_units = intrinsic.hidden_units
        target_params = _init_mlp(target_key, target_input_dim, hidden_units, target_output_dim)
        prior_params = _init_mlp(prior_key, target_input_dim, hidden_units, target_output_dim)
        predictor_params = _init_mlp(
            predictor_key,
            target_input_dim,
            hidden_units,
            target_output_dim,
        )
    optimizer = _optimizer(agent, intrinsic.learning_rate, intrinsic.optimizer)
    count_table_size = _intrinsic_count_table_size(
        intrinsic,
        num_actions=num_actions,
        oracle_state_space_size=oracle_state_space_size,
    )
    count_key_dim = _intrinsic_count_key_dim(input_dim, num_actions, intrinsic)
    return DqnIntrinsicState(
        target_params=target_params,
        prior_params=prior_params,
        predictor_params=predictor_params,
        opt_state=optimizer.init(predictor_params),
        reward_mean=jnp.asarray(0.0, dtype=jnp.float32),
        reward_var=jnp.asarray(1.0, dtype=jnp.float32),
        count_keys=jnp.zeros((count_table_size, count_key_dim), dtype=jnp.float32),
        counts=jnp.zeros((count_table_size,), dtype=jnp.float32),
        count_size=jnp.asarray(0, dtype=jnp.int32),
        count_overflow=jnp.asarray(False),
    )


def _sample_intrinsic_target(
    intrinsic: DqnIntrinsicConfig,
    key: jax.Array,
    target_dim: int,
) -> jax.Array:
    if intrinsic.kind != "cfn":
        return jnp.zeros((target_dim,), dtype=jnp.float32)
    targets = jax.random.bernoulli(key, p=0.5, shape=(target_dim,))
    return jnp.where(targets, 1.0, -1.0).astype(jnp.float32)


def _rnd_prediction_error(
    target_params,
    predictor_params,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    intrinsic_input = _conditioned_input(
        observations,
        actions,
        intrinsic.action_conditioning,
        num_actions,
    )
    target_features = jax.lax.stop_gradient(
        _select_conditioned_features(
            _apply_mlp(target_params, intrinsic_input, intrinsic.activation),
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
    )
    predictor_features = _select_conditioned_features(
        _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
        actions,
        intrinsic.action_conditioning,
        intrinsic.output_dim,
        num_actions,
    )
    prediction_error = jnp.mean(jnp.square(predictor_features - target_features), axis=-1)
    return prediction_error, intrinsic_input, target_features


def _cfn_outputs(
    prior_params,
    predictor_params,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    intrinsic_input = _conditioned_input(
        observations,
        actions,
        intrinsic.action_conditioning,
        num_actions,
    )
    prior_features = _maybe_tanh(
        _apply_mlp(prior_params, intrinsic_input, intrinsic.activation),
        intrinsic.cfn_final_tanh,
    )
    prior_features = jax.lax.stop_gradient(
        _select_conditioned_features(
            prior_features,
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
    )
    predictor_features = _maybe_tanh(
        _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
        intrinsic.cfn_final_tanh,
    )
    predictor_features = _select_conditioned_features(
        predictor_features,
        actions,
        intrinsic.action_conditioning,
        intrinsic.output_dim,
        num_actions,
    )
    if intrinsic.cfn_use_random_prior:
        coin_flips = predictor_features + intrinsic.cfn_prior_scale * prior_features
    else:
        coin_flips = predictor_features
    raw_bonus = jnp.mean(jnp.square(coin_flips), axis=-1)
    raw_bonus = raw_bonus**intrinsic.cfn_bonus_exponent
    return raw_bonus, intrinsic_input, prior_features, predictor_features, coin_flips


def _conditioned_input(
    observations: jax.Array,
    actions: jax.Array,
    mode: ActionConditioning,
    num_actions: int,
) -> jax.Array:
    if mode == "input":
        return jnp.concatenate(
            (observations, jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)),
            axis=-1,
        )
    if mode == "pair":
        action_features = jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)
        pair_features = observations[..., :, None] * action_features[..., None, :]
        return pair_features.reshape(observations.shape[0], observations.shape[-1] * num_actions)
    return observations


def _select_conditioned_features(
    features: jax.Array,
    actions: jax.Array,
    mode: ActionConditioning,
    output_dim: int,
    num_actions: int,
) -> jax.Array:
    if mode != "output":
        return features
    features = features.reshape(features.shape[0], num_actions, output_dim)
    return jnp.take_along_axis(features, actions[:, None, None], axis=1).squeeze(axis=1)


def _normalize_intrinsic_reward(
    intrinsic: DqnIntrinsicConfig,
    raw_bonus: jax.Array,
    reward_mean: jax.Array,
    reward_var: jax.Array,
) -> jax.Array:
    if intrinsic.intrinsic_reward_center:
        raw_bonus = raw_bonus - reward_mean
    reward_scale = jnp.sqrt(jnp.maximum(reward_var, intrinsic.intrinsic_reward_epsilon))
    normalized = raw_bonus / reward_scale
    if intrinsic.intrinsic_reward_clip is not None:
        normalized = jnp.clip(normalized, 0.0, intrinsic.intrinsic_reward_clip)
    # TODO: expose option of keeping negative rewards or shifting them to be non-negative
    # else:
    #     normalized = normalized - jnp.minimum(jnp.min(normalized), 0.0)
    return normalized


def _update_intrinsic_stats(
    intrinsic: DqnIntrinsicConfig,
    old_mean: jax.Array,
    old_var: jax.Array,
    raw_bonus: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_mean = jnp.mean(raw_bonus)
    batch_second_moment = jnp.mean(jnp.square(raw_bonus))
    old_second_moment = old_var + jnp.square(old_mean)
    decay = intrinsic.intrinsic_stats_decay
    new_mean = decay * old_mean + (1.0 - decay) * batch_mean
    new_second_moment = decay * old_second_moment + (1.0 - decay) * batch_second_moment
    new_var = jnp.maximum(
        new_second_moment - jnp.square(new_mean),
        intrinsic.intrinsic_reward_epsilon,
    )
    return new_mean, new_var


def _count_uses_oracle_tabular(intrinsic: DqnIntrinsicConfig) -> bool:
    return intrinsic.kind == "count" and intrinsic.count_key_mode == "oracle_tabular"


def _count_action_conditioned(intrinsic: DqnIntrinsicConfig) -> bool:
    return intrinsic.action_conditioning != "none"


def _count_direct_indices(
    state_ids: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    state_ids = state_ids.astype(jnp.int32)
    actions = actions.astype(jnp.int32)
    if not _count_action_conditioned(intrinsic):
        return state_ids
    return state_ids * num_actions + actions


def _count_direct_bonus(
    state: DqnIntrinsicState,
    state_ids: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    indices = _count_direct_indices(state_ids, actions, intrinsic, num_actions)
    indices = jnp.clip(indices, 0, state.counts.shape[0] - 1)
    counts = state.counts[indices]
    effective_counts = jnp.maximum(counts, intrinsic.count_min_count)
    return 1.0 / (effective_counts**intrinsic.count_bonus_exponent)


def _count_direct_bonus_for_all_actions(
    state: DqnIntrinsicState,
    state_ids: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    batch_size = state_ids.shape[0]
    actions = jnp.broadcast_to(
        jnp.arange(num_actions, dtype=jnp.int32),
        (batch_size, num_actions),
    )
    repeated_state_ids = jnp.broadcast_to(state_ids[:, None], (batch_size, num_actions))
    flat_bonus = _count_direct_bonus(
        state,
        repeated_state_ids.reshape(-1),
        actions.reshape(-1),
        intrinsic,
        num_actions,
    )
    return flat_bonus.reshape(batch_size, num_actions)


def _intrinsic_bonus_for_all_actions(
    state: DqnIntrinsicState,
    observations: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    batch_size = observations.shape[0]
    actions = jnp.broadcast_to(
        jnp.arange(num_actions, dtype=jnp.int32),
        (batch_size, num_actions),
    ).reshape(-1)
    repeated_observations = jnp.repeat(observations, num_actions, axis=0)
    bonuses = _intrinsic_bonus(
        state,
        repeated_observations,
        actions,
        intrinsic,
        num_actions,
    )
    return bonuses.reshape(batch_size, num_actions)


def _intrinsic_bonus(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    if intrinsic.kind == "none":
        return jnp.zeros_like(actions, dtype=jnp.float32)
    if intrinsic.kind == "rnd":
        raw_bonus, _intrinsic_input, _target_features = _rnd_prediction_error(
            state.target_params,
            state.predictor_params,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    elif intrinsic.kind == "cfn":
        raw_bonus, _intrinsic_input, _prior_features, _predictor_features, _coin_flips = (
            _cfn_outputs(
                state.prior_params,
                state.predictor_params,
                observations,
                actions,
                intrinsic,
                num_actions,
            )
        )
    elif intrinsic.kind == "simhash":
        return _simhash_raw_bonus(
            state,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    else:
        return _count_raw_bonus(
            state,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    return _normalize_intrinsic_reward(
        intrinsic,
        raw_bonus,
        state.reward_mean,
        state.reward_var,
    )


def _observe_intrinsic_transition(
    state: DqnIntrinsicState,
    observation: jax.Array,
    action: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
    *,
    state_id: jax.Array | None = None,
) -> DqnIntrinsicState:
    if intrinsic.kind not in {"count", "simhash"}:
        return state
    if _count_uses_oracle_tabular(intrinsic):
        if state_id is None:
            raise ValueError("oracle_tabular count requires state_id when observing transitions")
        return _observe_count_direct_transition(
            state,
            state_id,
            action,
            intrinsic,
            num_actions,
        )
    key = _intrinsic_count_keys(
        state,
        observation[None, :],
        action.reshape((1,)),
        intrinsic,
        num_actions,
    )[0]
    index, found = _count_lookup_one(state, key)
    has_capacity = state.count_size < state.counts.shape[0]
    should_insert = jnp.logical_and(~found, has_capacity)
    should_record = jnp.logical_or(found, has_capacity)
    write_index = jnp.where(found, index, state.count_size)
    safe_index = jnp.minimum(write_index, state.counts.shape[0] - 1).astype(jnp.int32)
    existing_key = state.count_keys[safe_index]
    count_keys = state.count_keys.at[safe_index].set(jnp.where(should_insert, key, existing_key))
    counts = state.counts.at[safe_index].add(should_record.astype(jnp.float32))
    return state._replace(
        count_keys=count_keys,
        counts=counts,
        count_size=state.count_size + should_insert.astype(jnp.int32),
        count_overflow=jnp.logical_or(
            state.count_overflow,
            jnp.logical_and(~found, ~has_capacity),
        ),
    )


def _observe_count_direct_transition(
    state: DqnIntrinsicState,
    state_id: jax.Array,
    action: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> DqnIntrinsicState:
    index = _count_direct_indices(
        state_id.reshape((1,)),
        action.reshape((1,)),
        intrinsic,
        num_actions,
    )[0]
    in_bounds = jnp.logical_and(index >= 0, index < state.counts.shape[0])
    safe_index = jnp.clip(index, 0, state.counts.shape[0] - 1)
    counts = state.counts.at[safe_index].add(in_bounds.astype(jnp.float32))
    return state._replace(
        counts=counts,
        count_overflow=jnp.logical_or(state.count_overflow, ~in_bounds),
    )


def _observe_intrinsic_transition_for_roles(
    intrinsic_state: DqnIntrinsicState,
    reward_intrinsic_state: DqnIntrinsicState,
    observation: jax.Array,
    action: jax.Array,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    shared_intrinsic: bool,
    num_actions: int,
    *,
    state_id: jax.Array | None = None,
) -> tuple[DqnIntrinsicState, DqnIntrinsicState]:
    intrinsic_state = _observe_intrinsic_transition(
        intrinsic_state,
        observation,
        action,
        knownness,
        num_actions,
        state_id=state_id,
    )
    if shared_intrinsic:
        return intrinsic_state, intrinsic_state
    reward_intrinsic_state = _observe_intrinsic_transition(
        reward_intrinsic_state,
        observation,
        action,
        intrinsic_reward,
        num_actions,
        state_id=state_id,
    )
    return intrinsic_state, reward_intrinsic_state


def _count_raw_bonus(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    keys = _count_keys(observations, actions, intrinsic, num_actions)
    indices, found = _count_lookup(state, keys)
    counts = jnp.where(found, state.counts[indices], 0.0)
    effective_counts = jnp.maximum(counts, intrinsic.count_min_count)
    return 1.0 / (effective_counts**intrinsic.count_bonus_exponent)


def _simhash_raw_bonus(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    keys = _simhash_keys(state, observations, actions, intrinsic, num_actions)
    indices, found = _count_lookup(state, keys)
    counts = jnp.where(found, state.counts[indices], 0.0)
    effective_counts = jnp.maximum(counts, intrinsic.simhash_min_count)
    return 1.0 / (effective_counts**intrinsic.simhash_bonus_exponent)


def _count_lookup(
    state: DqnIntrinsicState,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    valid = jnp.arange(state.counts.shape[0], dtype=jnp.int32) < state.count_size
    matches = jnp.all(state.count_keys[None, :, :] == keys[:, None, :], axis=-1)
    matches = jnp.logical_and(matches, valid[None, :])
    found = jnp.any(matches, axis=1)
    indices = jnp.argmax(matches.astype(jnp.int32), axis=1).astype(jnp.int32)
    return indices, found


def _count_lookup_one(
    state: DqnIntrinsicState,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    indices, found = _count_lookup(state, key[None, :])
    return indices[0], found[0]


def _count_keys(
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    observations = _count_observations_for_keys(observations, intrinsic)
    if intrinsic.action_conditioning == "none":
        return observations.astype(jnp.float32)
    if intrinsic.action_conditioning == "pair":
        return _conditioned_input(
            observations,
            actions,
            "pair",
            num_actions,
        ).astype(jnp.float32)
    return jnp.concatenate(
        (observations, jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)),
        axis=-1,
    ).astype(jnp.float32)


def _intrinsic_count_keys(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    if intrinsic.kind == "simhash":
        return _simhash_keys(state, observations, actions, intrinsic, num_actions)
    return _count_keys(observations, actions, intrinsic, num_actions)


def _simhash_keys(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    simhash_input = _simhash_input(observations, actions, intrinsic, num_actions)
    if intrinsic.simhash_mode == "learned":
        simhash_input = _apply_autoencoder_encoder(
            state.predictor_params,
            simhash_input,
            intrinsic.hidden_units,
            intrinsic.activation,
        )
    projections = _apply_mlp(state.target_params, simhash_input, "linear")
    return (projections >= 0.0).astype(jnp.float32)


def _simhash_input(
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    observations = _simhash_observations_for_keys(observations, intrinsic)
    return _conditioned_input(
        observations,
        actions,
        intrinsic.action_conditioning,
        num_actions,
    ).astype(jnp.float32)


def _count_observations_for_keys(
    observations: jax.Array,
    intrinsic: DqnIntrinsicConfig,
) -> jax.Array:
    if not intrinsic.count_ignore_empty_room_distractor:
        return observations
    return _ignore_empty_room_symbolic_distractor(observations)


def _simhash_observations_for_keys(
    observations: jax.Array,
    intrinsic: DqnIntrinsicConfig,
) -> jax.Array:
    if not intrinsic.simhash_ignore_empty_room_distractor:
        return observations
    return _ignore_empty_room_symbolic_distractor(observations)


def _ignore_empty_room_symbolic_distractor(observations: jax.Array) -> jax.Array:
    feature_dim = observations.shape[-1]
    if feature_dim % 3 != 0:
        return observations
    side = int(round(np.sqrt(feature_dim // 3)))
    if side * side * 3 != feature_dim:
        return observations

    grid = observations.reshape((*observations.shape[:-1], side, side, 3))
    is_normalized = jnp.max(jnp.abs(observations)) <= 1.0
    wall_entity = jnp.where(is_normalized, 2.0 / 255.0, 2.0)
    wall_colour = jnp.where(is_normalized, 5.0 / 255.0, 5.0)
    wall_mask = jnp.isclose(grid[..., 0], wall_entity)
    grid = grid.at[..., 1].set(jnp.where(wall_mask, wall_colour, grid[..., 1]))
    return grid.reshape(observations.shape)


def _count_key_dim(
    input_dim: int,
    num_actions: int,
    mode: ActionConditioning,
) -> int:
    if mode == "none":
        return input_dim
    if mode == "pair":
        return input_dim * num_actions
    return input_dim + num_actions


def _intrinsic_count_table_size(
    intrinsic: DqnIntrinsicConfig,
    *,
    num_actions: int,
    oracle_state_space_size: int | None,
) -> int:
    if intrinsic.kind == "count":
        if intrinsic.count_key_mode == "oracle_tabular":
            if oracle_state_space_size is None:
                raise ValueError("count_key_mode='oracle_tabular' requires oracle_state_space_size")
            action_factor = num_actions if _count_action_conditioned(intrinsic) else 1
            auto_size = oracle_state_space_size * action_factor
            return int(intrinsic.count_table_size or auto_size)
        if intrinsic.count_table_size < 1:
            raise ValueError("dense_exact count requires count_table_size >= 1")
        return intrinsic.count_table_size
    if intrinsic.kind == "simhash":
        return intrinsic.simhash_table_size
    return 1


def _intrinsic_count_key_dim(
    input_dim: int,
    num_actions: int,
    intrinsic: DqnIntrinsicConfig,
) -> int:
    if intrinsic.kind == "count":
        if intrinsic.count_key_mode == "oracle_tabular":
            return 1
        return _count_key_dim(input_dim, num_actions, intrinsic.action_conditioning)
    if intrinsic.kind == "simhash":
        return intrinsic.simhash_bits
    return 1


def _count_table_status(
    state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
) -> tuple[int | None, bool | None]:
    if intrinsic.kind not in {"count", "simhash"}:
        return None, None
    if _count_uses_oracle_tabular(intrinsic):
        entries = int(np.asarray(jax.device_get(jnp.sum(state.counts > 0))))
    else:
        entries = int(np.asarray(jax.device_get(state.count_size)))
    return (
        entries,
        bool(np.asarray(jax.device_get(state.count_overflow))),
    )


def _merge_count_table_status(
    knownness_entries: int | None,
    knownness_overflow: bool | None,
    reward_entries: int | None,
    reward_overflow: bool | None,
) -> tuple[int | None, bool | None]:
    entries = knownness_entries if knownness_entries is not None else reward_entries
    if knownness_overflow is None:
        return entries, reward_overflow
    if reward_overflow is None:
        return entries, knownness_overflow
    return entries, bool(knownness_overflow or reward_overflow)


def _handle_count_table_overflow(
    intrinsic: DqnIntrinsicConfig,
    overflow: bool | None,
) -> None:
    if intrinsic.kind not in {"count", "simhash"} or not overflow:
        return
    table_name = "simhash_table_size" if intrinsic.kind == "simhash" else "count_table_size"
    table_size = (
        intrinsic.simhash_table_size if intrinsic.kind == "simhash" else intrinsic.count_table_size
    )
    overflow_mode = (
        intrinsic.simhash_table_overflow
        if intrinsic.kind == "simhash"
        else intrinsic.count_table_overflow
    )
    message = (
        f"Count table exceeded {table_name}={table_size}; additional novel "
        f"count keys were not inserted. Increase {table_name} or set the "
        "table_overflow option to 'error' to fail runs."
    )
    if overflow_mode == "error":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _conditioned_input_dim(input_dim: int, num_actions: int, mode: ActionConditioning) -> int:
    if mode == "input":
        return input_dim + num_actions
    if mode == "pair":
        return input_dim * num_actions
    return input_dim


def _conditioned_output_dim(num_actions: int, output_dim: int, mode: ActionConditioning) -> int:
    if mode == "output":
        return num_actions * output_dim
    return output_dim


def _maybe_tanh(features: jax.Array, enabled: bool) -> jax.Array:
    if enabled:
        return jnp.tanh(features)
    return features


def _aux_params(
    state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
) -> dict[str, tuple[dict[str, jax.Array], ...]]:
    if intrinsic.kind == "rnd":
        return {
            "rnd_target": state.target_params,
            "rnd_predictor": state.predictor_params,
        }
    if intrinsic.kind == "cfn":
        return {
            "cfn_prior": state.prior_params,
            "cfn_predictor": state.predictor_params,
        }
    if intrinsic.kind == "simhash":
        params = {"simhash_projection": state.target_params}
        if intrinsic.simhash_mode == "learned":
            params["simhash_autoencoder"] = state.predictor_params
        return params
    return {}


def _aux_params_for_roles(
    knownness_state: DqnIntrinsicState,
    knownness: DqnIntrinsicConfig,
    reward_state: DqnIntrinsicState,
    intrinsic_reward: DqnIntrinsicConfig,
    *,
    shared_intrinsic: bool,
) -> dict[str, tuple[dict[str, jax.Array], ...]]:
    if shared_intrinsic:
        return _aux_params(knownness_state, knownness)
    if knownness.kind == "none":
        return _aux_params(reward_state, intrinsic_reward)
    if intrinsic_reward.kind == "none":
        return _aux_params(knownness_state, knownness)
    aux: dict[str, tuple[dict[str, jax.Array], ...]] = {}
    aux.update(
        {
            f"knownness_{name}": params
            for name, params in _aux_params(knownness_state, knownness).items()
        }
    )
    aux.update(
        {
            f"intrinsic_reward_{name}": params
            for name, params in _aux_params(reward_state, intrinsic_reward).items()
        }
    )
    return aux
