from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlflow_builtin.dqn.config import (
    DQN_AGENT_COMPONENT as DQN_AGENT_COMPONENT,
)
from rlflow_builtin.dqn.config import (
    DQN_RMAX_AGENT_COMPONENT as DQN_RMAX_AGENT_COMPONENT,
)
from rlflow_builtin.dqn.config import (
    ActionConditioning,
    DqnAgentConfig,
    DqnIntrinsicConfig,
    DqnReplayConfig,
)
from rlflow_builtin.dqn.config import (
    _canonicalize_action_conditioning as _canonicalize_action_conditioning,
)
from rlflow_builtin.dqn.config import (
    _simhash_mode as _simhash_mode,
)
from rlflow_builtin.dqn.config import (
    dqn_agent_config as dqn_agent_config,
)
from rlflow_builtin.dqn.config import (
    dqn_intrinsic_config as dqn_intrinsic_config,
)
from rlflow_builtin.dqn.config import (
    dqn_replay_config as dqn_replay_config,
)
from rlflow_builtin.dqn.networks import (
    _apply_autoencoder_encoder,
    _apply_mlp,
    _init_autoencoder,
    _init_mlp,
)
from rlflow_builtin.dqn.replay import (
    _initial_replay_state,
    _push_replay,
    _replay_arrays,
    _replay_intrinsic_target_dim,
    _resolve_replay_save_path,
    _sample_replay,
)
from rlflow_builtin.dqn.state import (
    DqnIntrinsicState,
    DqnRunResult,
    DqnTrainState,
    _DqnEnvironment,
)
from rlflow_builtin.tabular.environments import environment_config, initial_state, make_step_fn
from rlflow_builtin.tabular.types import RunnerConfig


def run_dqn_training(
    *,
    env_component: str,
    env_settings: dict[str, Any],
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    runner: RunnerConfig,
    intrinsic: DqnIntrinsicConfig | None = None,
    knownness: DqnIntrinsicConfig | None = None,
    intrinsic_reward: DqnIntrinsicConfig | None = None,
    shared_intrinsic: bool = False,
    run_dir: Path | None = None,
) -> DqnRunResult:
    none_intrinsic = dqn_intrinsic_config(None, None, agent)
    if intrinsic is not None and knownness is None and intrinsic_reward is None:
        if agent.algorithm == "dqn_rmax":
            knownness = intrinsic
        else:
            intrinsic_reward = intrinsic
    knownness = knownness or none_intrinsic
    intrinsic_reward = intrinsic_reward or none_intrinsic
    shared_intrinsic = bool(
        shared_intrinsic and knownness.kind != "none" and intrinsic_reward.kind != "none"
    )
    if agent.algorithm == "dqn_rmax" and knownness.kind == "none":
        raise ValueError("builtin.agent.dqn_rmax_jax requires a knownness_signal input")
    dqn_env = _make_dqn_environment(
        env_component,
        env_settings,
        normalize_observations=agent.normalize_observations,
    )
    seed = runner.seed + agent.seed
    key = jax.random.PRNGKey(seed)
    key, q_key, knownness_key, reward_intrinsic_key = jax.random.split(key, 4)

    q_optimizer = _optimizer(agent, agent.learning_rate, agent.optimizer)
    q_params = _init_mlp(q_key, dqn_env.input_dim, agent.hidden_units, dqn_env.num_actions)
    knownness_state = _initial_intrinsic_state(
        agent,
        knownness,
        dqn_env.input_dim,
        dqn_env.num_actions,
        knownness_key,
        oracle_state_space_size=dqn_env.oracle_state_space_size,
    )
    reward_intrinsic_state = (
        knownness_state
        if shared_intrinsic
        else _initial_intrinsic_state(
            agent,
            intrinsic_reward,
            dqn_env.input_dim,
            dqn_env.num_actions,
            reward_intrinsic_key,
            oracle_state_space_size=dqn_env.oracle_state_space_size,
        )
    )
    initial_state = DqnTrainState(
        params=q_params,
        target_params=_clone_params(q_params),
        opt_state=q_optimizer.init(q_params),
        intrinsic_state=knownness_state,
        reward_intrinsic_state=reward_intrinsic_state,
        replay_state=_initial_replay_state(
            replay.capacity,
            dqn_env.input_dim,
            _replay_intrinsic_target_dim(knownness),
            dqn_env.observation_shape,
            dqn_env.observation_dtype,
            reward_intrinsic_target_dim=_replay_intrinsic_target_dim(intrinsic_reward),
        ),
        key=key,
        global_step=jnp.asarray(0, dtype=jnp.int32),
        gradient_step=jnp.asarray(0, dtype=jnp.int32),
        intrinsic_gradient_step=jnp.asarray(0, dtype=jnp.int32),
        reward_intrinsic_gradient_step=jnp.asarray(0, dtype=jnp.int32),
    )
    intrinsic_optimizer = _optimizer(agent, knownness.learning_rate, knownness.optimizer)
    reward_intrinsic_optimizer = _optimizer(
        agent,
        intrinsic_reward.learning_rate,
        intrinsic_reward.optimizer,
    )

    if runner.train_steps is None:

        @jax.jit
        def train_scan(state: DqnTrainState):
            return jax.lax.scan(
                lambda carry, _: _train_episode(
                    carry,
                    dqn_env,
                    agent,
                    replay,
                    knownness,
                    intrinsic_reward,
                    shared_intrinsic,
                    q_optimizer,
                    intrinsic_optimizer,
                    reward_intrinsic_optimizer,
                    runner.max_episode_steps,
                ),
                state,
                xs=None,
                length=runner.train_episodes,
            )

        final_state, train_history = train_scan(initial_state)
    else:

        @jax.jit
        def train_scan(state: DqnTrainState):
            return _train_steps(
                state,
                dqn_env,
                agent,
                replay,
                knownness,
                intrinsic_reward,
                shared_intrinsic,
                q_optimizer,
                intrinsic_optimizer,
                reward_intrinsic_optimizer,
                runner.max_episode_steps,
                runner.train_steps or 0,
            )

        final_state, train_history = train_scan(initial_state)

    knownness_count_entries, knownness_count_overflow = _count_table_status(
        final_state.intrinsic_state,
        knownness,
    )
    reward_count_entries, reward_count_overflow = (
        (knownness_count_entries, knownness_count_overflow)
        if shared_intrinsic
        else _count_table_status(final_state.reward_intrinsic_state, intrinsic_reward)
    )
    count_entries, count_overflow = _merge_count_table_status(
        knownness_count_entries,
        knownness_count_overflow,
        reward_count_entries,
        reward_count_overflow,
    )
    _handle_count_table_overflow(knownness, knownness_count_overflow)
    if not shared_intrinsic:
        _handle_count_table_overflow(intrinsic_reward, reward_count_overflow)

    if runner.eval_episodes > 0:

        @jax.jit
        def eval_scan(eval_key: jax.Array):
            return jax.lax.scan(
                lambda carry, _: _eval_episode(
                    carry,
                    final_state.params,
                    final_state.intrinsic_state,
                    dqn_env,
                    agent,
                    knownness,
                    runner.max_episode_steps,
                ),
                eval_key,
                xs=None,
                length=runner.eval_episodes,
            )

        _, eval_history = eval_scan(jax.random.PRNGKey(seed + 10000))
        eval_returns, eval_discounted_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_discounted_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_discounted_returns, train_lengths, train_losses = train_history
    train_returns_np = np.asarray(train_returns)
    train_discounted_returns_np = np.asarray(train_discounted_returns)
    train_lengths_np = np.asarray(train_lengths)
    train_losses_np = np.asarray(train_losses)
    if runner.train_steps is not None:
        episode_count = int(np.count_nonzero(train_lengths_np))
        train_returns_np = train_returns_np[:episode_count]
        train_discounted_returns_np = train_discounted_returns_np[:episode_count]
        train_lengths_np = train_lengths_np[:episode_count]
        train_losses_np = train_losses_np[:episode_count]
    replay_arrays = (
        _replay_arrays(
            final_state.replay_state,
            knownness,
            intrinsic_reward,
            shared_intrinsic=shared_intrinsic,
        )
        if replay.save_dataset_path
        else None
    )
    if replay.save_dataset_path and run_dir is not None and replay_arrays is not None:
        save_path = _resolve_replay_save_path(replay.save_dataset_path, run_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **replay_arrays)

    return DqnRunResult(
        params=final_state.params,
        aux_params=_aux_params_for_roles(
            final_state.intrinsic_state,
            knownness,
            final_state.reward_intrinsic_state,
            intrinsic_reward,
            shared_intrinsic=shared_intrinsic,
        ),
        train_returns=train_returns_np,
        train_discounted_returns=train_discounted_returns_np,
        train_lengths=train_lengths_np,
        train_losses=train_losses_np,
        eval_returns=np.asarray(eval_returns),
        eval_discounted_returns=np.asarray(eval_discounted_returns),
        eval_lengths=np.asarray(eval_lengths),
        source_observation_shape=dqn_env.observation_shape,
        source_observation_dtype=dqn_env.observation_dtype,
        input_dim=dqn_env.input_dim,
        num_actions=dqn_env.num_actions,
        replay_arrays=replay_arrays,
        count_table_entries=count_entries,
        count_table_overflow=count_overflow,
    )


def _train_episode(
    state: DqnTrainState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    shared_intrinsic: bool,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    reward_intrinsic_optimizer: optax.GradientTransformation,
    max_episode_steps: int,
) -> tuple[DqnTrainState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(state.key)
    env_state = dqn_env.reset(reset_key)
    source_observation = dqn_env.observation(env_state)
    observation = dqn_env.encode(source_observation)
    step_carry = (
        state._replace(key=key),
        env_state,
        observation,
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def step_fn(carry, _):
        (
            train_state,
            env_state,
            observation,
            done,
            episode_return,
            discounted_return,
            discount_power,
            episode_length,
            loss_sum,
        ) = carry

        def inactive(active_carry):
            return active_carry, jnp.asarray(0.0, dtype=jnp.float32)

        def active(active_carry):
            (
                train_state,
                env_state,
                observation,
                _done,
                episode_return,
                discounted_return,
                discount_power,
                episode_length,
                loss_sum,
            ) = active_carry
            key, action_key, env_key, replay_key, update_key = jax.random.split(
                train_state.key,
                5,
            )
            state_id = _oracle_state_id_for_intrinsics(
                dqn_env,
                env_state,
                knownness,
                intrinsic_reward,
            )
            action = _select_action(
                agent,
                train_state.params,
                train_state.intrinsic_state,
                knownness,
                observation,
                action_key,
                dqn_env.num_actions,
                train_state.global_step,
                training=True,
                state_id=state_id,
            )
            next_env_state = dqn_env.step(env_state, action, env_key)
            next_source_observation = dqn_env.observation(next_env_state)
            next_observation = dqn_env.encode(next_source_observation)
            next_state_id = _oracle_state_id_for_intrinsics(
                dqn_env,
                next_env_state,
                knownness,
                intrinsic_reward,
            )
            reward = dqn_env.reward(next_env_state).astype(jnp.float32)
            terminal = dqn_env.done(next_env_state)
            knownness_target_key, reward_target_key = jax.random.split(replay_key)
            intrinsic_target = _sample_intrinsic_target(
                knownness,
                knownness_target_key,
                train_state.replay_state.intrinsic_targets.shape[-1],
            )
            reward_intrinsic_target = (
                intrinsic_target
                if shared_intrinsic
                else _sample_intrinsic_target(
                    intrinsic_reward,
                    reward_target_key,
                    train_state.replay_state.reward_intrinsic_targets.shape[-1],
                )
            )
            replay_state = _push_replay(
                train_state.replay_state,
                observation,
                dqn_env.observation(env_state),
                action,
                reward,
                next_observation,
                next_source_observation,
                terminal,
                intrinsic_target,
                state_id,
                next_state_id,
                reward_intrinsic_target,
            )
            intrinsic_state, reward_intrinsic_state = _observe_intrinsic_transition_for_roles(
                train_state.intrinsic_state,
                train_state.reward_intrinsic_state,
                observation,
                action,
                knownness,
                intrinsic_reward,
                shared_intrinsic,
                dqn_env.num_actions,
                state_id=state_id,
            )
            train_state = train_state._replace(
                replay_state=replay_state,
                intrinsic_state=intrinsic_state,
                reward_intrinsic_state=reward_intrinsic_state,
                key=update_key,
            )
            should_update = jnp.logical_and(
                replay_state.size >= replay.min_size,
                train_state.global_step % agent.update_frequency == 0,
            )
            train_state, step_loss = jax.lax.cond(
                should_update,
                lambda update_state: _replay_updates(
                    update_state,
                    agent,
                    replay,
                    knownness,
                    intrinsic_reward,
                    shared_intrinsic,
                    q_optimizer,
                    intrinsic_optimizer,
                    reward_intrinsic_optimizer,
                    dqn_env.num_actions,
                ),
                lambda update_state: (
                    update_state,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                train_state,
            )
            train_state = train_state._replace(
                global_step=train_state.global_step + 1,
            )
            return (
                train_state,
                next_env_state,
                next_observation,
                terminal,
                episode_return + reward,
                discounted_return + discount_power * reward,
                discount_power * agent.discount,
                episode_length + 1,
                loss_sum + step_loss,
            ), step_loss

        return jax.lax.cond(done, inactive, active, carry)

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=max_episode_steps,
    )
    (
        state,
        _,
        _,
        _,
        episode_return,
        discounted_return,
        _discount_power,
        episode_length,
        loss_sum,
    ) = step_carry
    mean_loss = loss_sum / jnp.maximum(episode_length, 1)
    return state, (episode_return, discounted_return, episode_length, mean_loss)


def _train_steps(
    state: DqnTrainState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    shared_intrinsic: bool,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    reward_intrinsic_optimizer: optax.GradientTransformation,
    max_episode_steps: int,
    train_steps: int,
) -> tuple[DqnTrainState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(state.key)
    env_state = dqn_env.reset(reset_key)
    source_observation = dqn_env.observation(env_state)
    observation = dqn_env.encode(source_observation)
    train_returns = jnp.zeros((train_steps,), dtype=jnp.float32)
    train_discounted_returns = jnp.zeros((train_steps,), dtype=jnp.float32)
    train_lengths = jnp.zeros((train_steps,), dtype=jnp.int32)
    train_losses = jnp.zeros((train_steps,), dtype=jnp.float32)
    step_carry = (
        state._replace(key=key),
        env_state,
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        train_returns,
        train_discounted_returns,
        train_lengths,
        train_losses,
    )

    def write_episode(
        episode_index: jax.Array,
        returns: jax.Array,
        discounted_returns: jax.Array,
        lengths: jax.Array,
        losses: jax.Array,
        episode_return: jax.Array,
        discounted_return: jax.Array,
        episode_length: jax.Array,
        loss_sum: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        mean_loss = loss_sum / jnp.maximum(episode_length, 1)
        return (
            episode_index + 1,
            returns.at[episode_index].set(episode_return),
            discounted_returns.at[episode_index].set(discounted_return),
            lengths.at[episode_index].set(episode_length),
            losses.at[episode_index].set(mean_loss),
        )

    def step_fn(carry, _):
        (
            train_state,
            env_state,
            observation,
            episode_return,
            discounted_return,
            discount_power,
            episode_length,
            loss_sum,
            episode_index,
            returns,
            discounted_returns,
            lengths,
            losses,
        ) = carry
        _key, action_key, env_key, replay_key, update_key, reset_key = jax.random.split(
            train_state.key,
            6,
        )
        state_id = _oracle_state_id_for_intrinsics(
            dqn_env,
            env_state,
            knownness,
            intrinsic_reward,
        )
        action = _select_action(
            agent,
            train_state.params,
            train_state.intrinsic_state,
            knownness,
            observation,
            action_key,
            dqn_env.num_actions,
            train_state.global_step,
            training=True,
            state_id=state_id,
        )
        next_env_state = dqn_env.step(env_state, action, env_key)
        next_source_observation = dqn_env.observation(next_env_state)
        next_observation = dqn_env.encode(next_source_observation)
        next_state_id = _oracle_state_id_for_intrinsics(
            dqn_env,
            next_env_state,
            knownness,
            intrinsic_reward,
        )
        reward = dqn_env.reward(next_env_state).astype(jnp.float32)
        terminal = dqn_env.done(next_env_state)
        knownness_target_key, reward_target_key = jax.random.split(replay_key)
        intrinsic_target = _sample_intrinsic_target(
            knownness,
            knownness_target_key,
            train_state.replay_state.intrinsic_targets.shape[-1],
        )
        reward_intrinsic_target = (
            intrinsic_target
            if shared_intrinsic
            else _sample_intrinsic_target(
                intrinsic_reward,
                reward_target_key,
                train_state.replay_state.reward_intrinsic_targets.shape[-1],
            )
        )
        replay_state = _push_replay(
            train_state.replay_state,
            observation,
            dqn_env.observation(env_state),
            action,
            reward,
            next_observation,
            next_source_observation,
            terminal,
            intrinsic_target,
            state_id,
            next_state_id,
            reward_intrinsic_target,
        )
        intrinsic_state, reward_intrinsic_state = _observe_intrinsic_transition_for_roles(
            train_state.intrinsic_state,
            train_state.reward_intrinsic_state,
            observation,
            action,
            knownness,
            intrinsic_reward,
            shared_intrinsic,
            dqn_env.num_actions,
            state_id=state_id,
        )
        train_state = train_state._replace(
            replay_state=replay_state,
            intrinsic_state=intrinsic_state,
            reward_intrinsic_state=reward_intrinsic_state,
            key=update_key,
        )
        should_update = jnp.logical_and(
            replay_state.size >= replay.min_size,
            train_state.global_step % agent.update_frequency == 0,
        )
        train_state, step_loss = jax.lax.cond(
            should_update,
            lambda update_state: _replay_updates(
                update_state,
                agent,
                replay,
                knownness,
                intrinsic_reward,
                shared_intrinsic,
                q_optimizer,
                intrinsic_optimizer,
                reward_intrinsic_optimizer,
                dqn_env.num_actions,
            ),
            lambda update_state: (
                update_state,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            train_state,
        )
        train_state = train_state._replace(global_step=train_state.global_step + 1)
        next_episode_return = episode_return + reward
        next_discounted_return = discounted_return + discount_power * reward
        next_discount_power = discount_power * agent.discount
        next_episode_length = episode_length + 1
        next_loss_sum = loss_sum + step_loss
        episode_done = jnp.logical_or(terminal, next_episode_length >= max_episode_steps)

        def finish_episode(args):
            (
                train_state,
                _next_env_state,
                _next_observation,
                next_episode_return,
                next_discounted_return,
                _next_discount_power,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                discounted_returns,
                lengths,
                losses,
                reset_key,
            ) = args
            episode_index, returns, discounted_returns, lengths, losses = write_episode(
                episode_index,
                returns,
                discounted_returns,
                lengths,
                losses,
                next_episode_return,
                next_discounted_return,
                next_episode_length,
                next_loss_sum,
            )
            reset_env_state = dqn_env.reset(reset_key)
            reset_source_observation = dqn_env.observation(reset_env_state)
            reset_observation = dqn_env.encode(reset_source_observation)
            return (
                train_state,
                reset_env_state,
                reset_observation,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
                episode_index,
                returns,
                discounted_returns,
                lengths,
                losses,
            )

        def continue_episode(args):
            (
                train_state,
                next_env_state,
                next_observation,
                next_episode_return,
                next_discounted_return,
                next_discount_power,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                discounted_returns,
                lengths,
                losses,
                _reset_key,
            ) = args
            return (
                train_state,
                next_env_state,
                next_observation,
                next_episode_return,
                next_discounted_return,
                next_discount_power,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                discounted_returns,
                lengths,
                losses,
            )

        return (
            jax.lax.cond(
                episode_done,
                finish_episode,
                continue_episode,
                (
                    train_state,
                    next_env_state,
                    next_observation,
                    next_episode_return,
                    next_discounted_return,
                    next_discount_power,
                    next_episode_length,
                    next_loss_sum,
                    episode_index,
                    returns,
                    discounted_returns,
                    lengths,
                    losses,
                    reset_key,
                ),
            ),
            None,
        )

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=train_steps,
    )
    (
        state,
        _env_state,
        _observation,
        episode_return,
        discounted_return,
        _discount_power,
        episode_length,
        loss_sum,
        episode_index,
        returns,
        discounted_returns,
        lengths,
        losses,
    ) = step_carry

    def write_partial(args):
        episode_index, returns, discounted_returns, lengths, losses = write_episode(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            episode_return,
            discounted_return,
            episode_length,
            loss_sum,
        )
        return episode_index, returns, discounted_returns, lengths, losses

    def skip_partial(args):
        return args

    _episode_index, returns, discounted_returns, lengths, losses = jax.lax.cond(
        episode_length > 0,
        write_partial,
        skip_partial,
        (episode_index, returns, discounted_returns, lengths, losses),
    )
    return state, (returns, discounted_returns, lengths, losses)


def _replay_updates(
    state: DqnTrainState,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    shared_intrinsic: bool,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    reward_intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
) -> tuple[DqnTrainState, jax.Array]:
    intrinsic_updates = _intrinsic_updates_per_step(replay)
    if knownness.kind == "none" and intrinsic_reward.kind == "none":
        intrinsic_updates = 0
    q_network_updates = _q_network_updates_per_step(replay)

    def intrinsic_update_step(carry, _):
        train_state, loss_sum = carry
        key, sample_key = jax.random.split(train_state.key)
        batch = _sample_replay(train_state.replay_state, sample_key, replay.batch_size)
        train_state = train_state._replace(key=key)
        train_state, loss = _intrinsic_update_roles_from_batch(
            train_state,
            batch,
            knownness,
            intrinsic_reward,
            shared_intrinsic,
            intrinsic_optimizer,
            reward_intrinsic_optimizer,
            num_actions,
        )
        return (train_state, loss_sum + loss), loss

    def q_update_step(carry, _):
        train_state, loss_sum = carry
        key, sample_key = jax.random.split(train_state.key)
        q_loss_key, key = jax.random.split(key)
        batch = _sample_replay(train_state.replay_state, sample_key, replay.batch_size)
        train_state = train_state._replace(key=key)
        train_state, loss = _q_update_from_batch(
            train_state,
            batch,
            agent,
            knownness,
            intrinsic_reward,
            q_optimizer,
            num_actions,
            q_loss_key,
        )
        return (train_state, loss_sum + loss), loss

    zero_loss = jnp.asarray(0.0, dtype=jnp.float32)
    intrinsic_loss_sum = zero_loss
    if intrinsic_updates > 0:
        (state, intrinsic_loss_sum), _ = jax.lax.scan(
            intrinsic_update_step,
            (state, zero_loss),
            xs=None,
            length=intrinsic_updates,
        )
    q_loss_sum = zero_loss
    if q_network_updates > 0:
        (state, q_loss_sum), _ = jax.lax.scan(
            q_update_step,
            (state, zero_loss),
            xs=None,
            length=q_network_updates,
        )
    intrinsic_loss = intrinsic_loss_sum / intrinsic_updates if intrinsic_updates > 0 else zero_loss
    q_loss = q_loss_sum / q_network_updates if q_network_updates > 0 else zero_loss
    return state, intrinsic_loss + q_loss


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


def _q_update_from_batch(
    state: DqnTrainState,
    batch: dict[str, jax.Array],
    agent: DqnAgentConfig,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
    q_optimizer: optax.GradientTransformation,
    num_actions: int,
    key: jax.Array,
) -> tuple[DqnTrainState, jax.Array]:
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_observations = batch["next_observations"]
    terminals = batch["terminals"]
    rmax_known_mask, rmax_next_unknown = _rmax_batch_masks(
        state.intrinsic_state,
        knownness,
        agent,
        batch,
        num_actions,
    )
    intrinsic_rewards = _batch_intrinsic_rewards(
        state.reward_intrinsic_state,
        batch,
        intrinsic_reward,
        num_actions,
    )
    total_rewards = rewards + intrinsic_reward.intrinsic_reward_scale * jax.lax.stop_gradient(
        intrinsic_rewards
    )

    def q_loss_fn(params):
        return _q_loss(
            agent,
            params,
            state.target_params,
            observations,
            actions,
            total_rewards,
            next_observations,
            terminals,
            rmax_known_mask,
            rmax_next_unknown,
            key,
        )

    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(state.params)
    q_updates, opt_state = q_optimizer.update(q_grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, q_updates)
    gradient_step = state.gradient_step + 1
    should_update_target = gradient_step % agent.target_update_frequency == 0
    target_params = _maybe_hard_update(params, state.target_params, should_update_target)
    return (
        state._replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            gradient_step=gradient_step,
        ),
        q_loss,
    )


def _intrinsic_updates_per_step(replay: DqnReplayConfig) -> int:
    if replay.intrinsic_updates_per_step is None:
        return replay.updates_per_step
    return replay.intrinsic_updates_per_step


def _q_network_updates_per_step(replay: DqnReplayConfig) -> int:
    if replay.q_network_updates_per_step is None:
        return replay.updates_per_step
    return replay.q_network_updates_per_step


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


def _eval_episode(
    key: jax.Array,
    params: tuple[dict[str, jax.Array], ...],
    intrinsic_state: DqnIntrinsicState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    intrinsic: DqnIntrinsicConfig,
    max_episode_steps: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(key)
    env_state = dqn_env.reset(reset_key)
    observation = dqn_env.encode(dqn_env.observation(env_state))
    step_carry = (
        key,
        env_state,
        observation,
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step_fn(carry, _):
        (
            key,
            env_state,
            observation,
            done,
            episode_return,
            discounted_return,
            discount_power,
            episode_length,
        ) = carry

        def inactive(active_carry):
            return active_carry, None

        def active(active_carry):
            (
                key,
                env_state,
                observation,
                _done,
                episode_return,
                discounted_return,
                discount_power,
                episode_length,
            ) = active_carry
            key, action_key, env_key = jax.random.split(key, 3)
            state_id = _oracle_state_id(dqn_env, env_state, intrinsic)
            action = _select_action(
                agent,
                params,
                intrinsic_state,
                intrinsic,
                observation,
                action_key,
                dqn_env.num_actions,
                jnp.asarray(0, dtype=jnp.int32),
                training=False,
                state_id=state_id,
            )
            next_env_state = dqn_env.step(env_state, action, env_key)
            reward = dqn_env.reward(next_env_state).astype(jnp.float32)
            terminal = dqn_env.done(next_env_state)
            next_observation = dqn_env.encode(dqn_env.observation(next_env_state))
            return (
                key,
                next_env_state,
                next_observation,
                terminal,
                episode_return + reward,
                discounted_return + discount_power * reward,
                discount_power * agent.discount,
                episode_length + 1,
            ), None

        return jax.lax.cond(done, inactive, active, carry)

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=max_episode_steps,
    )
    key, _, _, _, episode_return, discounted_return, _discount_power, episode_length = step_carry
    return key, (episode_return, discounted_return, episode_length)


def _make_dqn_environment(
    env_component: str,
    env_settings: dict[str, Any],
    *,
    normalize_observations: bool = False,
) -> _DqnEnvironment:
    if env_component == "navix.env.grid":
        from rlflow_builtin.environments.navix import (
            _state_space_size,
            _validate_spec,
            create_navix_environment,
            tabular_observation,
        )

        settings = _coerce_navix_settings(env_settings)
        spec = _validate_spec(
            settings["env_name"],
            settings["size"],
            settings["layout"],
            settings["observation_mode"],
            settings["action_set"],
            settings.get("symbolic_distractor", "none"),
        )
        env = create_navix_environment(**settings)
        shape = tuple(env.observation_space.shape)
        input_dim = _input_dim_from_space(env.observation_space)
        observation_dtype = np.dtype(env.observation_space.dtype)
        is_scalar = shape in {(), (1,)}
        is_integer = np.issubdtype(observation_dtype, np.integer)
        normalization_scale = _integer_observation_scale(observation_dtype)

        def encode(observation):
            observation = jnp.asarray(observation)
            if is_scalar:
                return jax.nn.one_hot(
                    observation.reshape(()).astype(jnp.int32),
                    input_dim,
                    dtype=jnp.float32,
                )
            encoded = observation.astype(jnp.float32).reshape(-1)
            if normalize_observations and is_integer and normalization_scale is not None:
                encoded = encoded / normalization_scale
            return encoded

        return _DqnEnvironment(
            observation_shape=shape,
            observation_dtype=str(observation_dtype),
            input_dim=input_dim,
            num_actions=int(env.action_space.n),
            reset=lambda key: env.reset(key),
            step=lambda timestep, action, _key: env.step(timestep, action),
            observation=lambda timestep: timestep.observation,
            reward=lambda timestep: timestep.reward,
            done=_timestep_done,
            encode=encode,
            oracle_state_id=lambda timestep: tabular_observation(timestep.state, spec=spec),
            oracle_state_space_size=_state_space_size(spec),
        )

    tabular_env = environment_config(env_component, env_settings)
    step_fn = make_step_fn(tabular_env)

    def reset(key):
        return (
            initial_state(tabular_env, key),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
        )

    def step(state, action, key):
        current_state = state[0] if isinstance(state, tuple) else state
        return step_fn(current_state, action, key)

    def encode(observation):
        return jax.nn.one_hot(
            jnp.asarray(observation).reshape(()).astype(jnp.int32),
            tabular_env.num_states,
            dtype=jnp.float32,
        )

    return _DqnEnvironment(
        observation_shape=(),
        observation_dtype="int32",
        input_dim=tabular_env.num_states,
        num_actions=tabular_env.num_actions,
        reset=reset,
        step=step,
        observation=lambda state: state[0] if isinstance(state, tuple) else state,
        reward=lambda state: (
            state[1] if isinstance(state, tuple) else jnp.asarray(0.0, dtype=jnp.float32)
        ),
        done=lambda state: state[2] if isinstance(state, tuple) else jnp.asarray(False),
        encode=encode,
        oracle_state_id=lambda state: state[0] if isinstance(state, tuple) else state,
        oracle_state_space_size=tabular_env.num_states,
    )


def _oracle_state_id(
    dqn_env: _DqnEnvironment,
    env_state: Any,
    intrinsic: DqnIntrinsicConfig,
) -> jax.Array:
    if _count_uses_oracle_tabular(intrinsic):
        if dqn_env.oracle_state_id is None:
            raise ValueError(
                "count_key_mode='oracle_tabular' requires an environment oracle_state_id"
            )
        return jnp.asarray(dqn_env.oracle_state_id(env_state), dtype=jnp.int32)
    return jnp.asarray(0, dtype=jnp.int32)


def _oracle_state_id_for_intrinsics(
    dqn_env: _DqnEnvironment,
    env_state: Any,
    knownness: DqnIntrinsicConfig,
    intrinsic_reward: DqnIntrinsicConfig,
) -> jax.Array:
    if _count_uses_oracle_tabular(knownness) or _count_uses_oracle_tabular(intrinsic_reward):
        if dqn_env.oracle_state_id is None:
            raise ValueError(
                "count_key_mode='oracle_tabular' requires an environment oracle_state_id"
            )
        return jnp.asarray(dqn_env.oracle_state_id(env_state), dtype=jnp.int32)
    return jnp.asarray(0, dtype=jnp.int32)


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


def _q_loss(
    agent: DqnAgentConfig,
    params,
    target_params,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
    known_mask: jax.Array,
    next_unknown_any: jax.Array,
    key: jax.Array,
) -> jax.Array:
    q_values = _apply_mlp(params, observations, agent.activation)
    selected_q = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
    if agent.double_q:
        next_online_q = _apply_mlp(params, next_observations, agent.activation)
        next_actions = _max_actions(jax.lax.stop_gradient(next_online_q), key)[:, None]
        next_target_q = _apply_mlp(target_params, next_observations, agent.activation)
        next_q = jnp.take_along_axis(next_target_q, next_actions, axis=1).squeeze(-1)
    else:
        next_q = jnp.max(_apply_mlp(target_params, next_observations, agent.activation), axis=1)
    if agent.algorithm == "dqn_rmax":
        next_q = jnp.where(next_unknown_any, agent.rmax_update_v_max, next_q)
    target = rewards + agent.discount * next_q * (1.0 - terminals)
    td_error = selected_q - jax.lax.stop_gradient(target)
    losses = _td_loss(td_error, agent.loss_type, agent.huber_delta)
    if agent.algorithm == "dqn_rmax":
        weights = known_mask.astype(jnp.float32)
        return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.mean(losses)


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


def _td_loss(td_error: jax.Array, loss_type: str, huber_delta: float) -> jax.Array:
    if loss_type == "mse":
        return jnp.square(td_error)
    abs_error = jnp.abs(td_error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + huber_delta * linear


def _optimizer(
    agent: DqnAgentConfig,
    learning_rate: float,
    optimizer_name: str,
) -> optax.GradientTransformation:
    if optimizer_name == "sgd":
        base = optax.sgd(learning_rate, momentum=agent.optimizer_momentum)
    elif optimizer_name == "rmsprop":
        base = optax.rmsprop(
            learning_rate,
            decay=agent.optimizer_decay,
            eps=agent.optimizer_epsilon,
            momentum=agent.optimizer_momentum,
            centered=agent.optimizer_centered,
        )
    else:
        base = optax.adamw(
            learning_rate,
            b1=agent.optimizer_beta1,
            b2=agent.optimizer_beta2,
            eps=agent.optimizer_epsilon,
            weight_decay=agent.optimizer_weight_decay,
        )
    if agent.max_grad_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(agent.max_grad_norm), base)
    return base


def _input_dim_from_space(space: Any) -> int:
    shape = tuple(space.shape)
    if shape in {(), (1,)} and hasattr(space, "n"):
        return int(space.n)
    return int(np.prod(np.asarray(shape)))


def _integer_observation_scale(dtype: np.dtype) -> float | None:
    if not np.issubdtype(dtype, np.integer):
        return None
    info = np.iinfo(dtype)
    return float(info.max) if info.max > 0 else None


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


def _clone_params(params):
    return jax.tree_util.tree_map(lambda item: jnp.array(item, copy=True), params)


def _maybe_hard_update(source, target, should_update: jax.Array):
    return jax.tree_util.tree_map(
        lambda source_leaf, target_leaf: jnp.where(should_update, source_leaf, target_leaf),
        source,
        target,
    )


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


def _coerce_navix_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_name": settings["env_name"],
        "size": int(settings["size"]),
        "layout": settings["layout"],
        "observation_mode": settings["observation_mode"],
        "action_set": settings["action_set"],
        "max_steps": settings["max_steps"],
        "symbolic_distractor": settings.get("symbolic_distractor", "none"),
    }


def _timestep_done(timestep: Any) -> jax.Array:
    if hasattr(timestep, "is_done"):
        return timestep.is_done()
    return jnp.logical_or(timestep.is_termination(), timestep.is_truncation())
