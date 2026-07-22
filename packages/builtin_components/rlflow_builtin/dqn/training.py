from __future__ import annotations

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
from rlflow_builtin.dqn.intrinsic import (
    _aux_params_for_roles,
    _batch_intrinsic_rewards,
    _count_table_status,
    _count_uses_oracle_tabular,
    _handle_count_table_overflow,
    _initial_intrinsic_state,
    _intrinsic_update_roles_from_batch,
    _merge_count_table_status,
    _observe_intrinsic_transition_for_roles,
    _sample_intrinsic_target,
)
from rlflow_builtin.dqn.intrinsic import _cfn_outputs as _cfn_outputs
from rlflow_builtin.dqn.intrinsic import _cfn_update as _cfn_update
from rlflow_builtin.dqn.intrinsic import _conditioned_input as _conditioned_input
from rlflow_builtin.dqn.intrinsic import _conditioned_input_dim as _conditioned_input_dim
from rlflow_builtin.dqn.intrinsic import _conditioned_output_dim as _conditioned_output_dim
from rlflow_builtin.dqn.intrinsic import _count_direct_bonus as _count_direct_bonus
from rlflow_builtin.dqn.intrinsic import _count_keys as _count_keys
from rlflow_builtin.dqn.intrinsic import _count_raw_bonus as _count_raw_bonus
from rlflow_builtin.dqn.intrinsic import _normalize_intrinsic_reward as _normalize_intrinsic_reward
from rlflow_builtin.dqn.intrinsic import (
    _observe_intrinsic_transition as _observe_intrinsic_transition,
)
from rlflow_builtin.dqn.intrinsic import _rnd_prediction_error as _rnd_prediction_error
from rlflow_builtin.dqn.intrinsic import _rnd_update as _rnd_update
from rlflow_builtin.dqn.intrinsic import (
    _select_conditioned_features as _select_conditioned_features,
)
from rlflow_builtin.dqn.intrinsic import _simhash_raw_bonus as _simhash_raw_bonus
from rlflow_builtin.dqn.intrinsic import _simhash_update as _simhash_update
from rlflow_builtin.dqn.networks import (
    _apply_mlp,
    _init_mlp,
    _optimizer,
)
from rlflow_builtin.dqn.policies import _epsilon as _epsilon
from rlflow_builtin.dqn.policies import _max_actions, _rmax_batch_masks, _select_action
from rlflow_builtin.dqn.policies import _rmax_action as _rmax_action
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


def _td_loss(td_error: jax.Array, loss_type: str, huber_delta: float) -> jax.Array:
    if loss_type == "mse":
        return jnp.square(td_error)
    abs_error = jnp.abs(td_error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + huber_delta * linear


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


def _clone_params(params):
    return jax.tree_util.tree_map(lambda item: jnp.array(item, copy=True), params)


def _maybe_hard_update(source, target, should_update: jax.Array):
    return jax.tree_util.tree_map(
        lambda source_leaf, target_leaf: jnp.where(should_update, source_leaf, target_leaf),
        source,
        target,
    )


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
