from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.tabular.agents import apply_td_update
from rlflow_builtin.tabular.buffers import initial_replay_buffer, no_buffer_config, push_transition, replay_dataset_arrays, sample_batch
from rlflow_builtin.tabular.environments import initial_state, make_step_fn
from rlflow_builtin.tabular.policies import greedy, select_action
from rlflow_builtin.tabular.types import (
    AgentConfig,
    BufferConfig,
    EnvironmentConfig,
    PolicyConfig,
    ReplayBufferState,
    RunnerConfig,
    TabularDataset,
    TabularRunResult,
    TransitionBatch,
)


class RMaxModelState(NamedTuple):
    q_table: jax.Array
    action_counts: jax.Array
    model_counts: jax.Array
    reward_sums: jax.Array
    transition_counts: jax.Array


def run_tabular_training(
    agent: AgentConfig,
    policy: PolicyConfig | None,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
    replay_buffer: BufferConfig | None = None,
) -> TabularRunResult:
    replay_buffer = replay_buffer or no_buffer_config()
    if agent.algorithm == "rmax":
        if replay_buffer.enabled:
            raise ValueError("builtin.agent.rmax_tabular does not support replay buffers")
        if environment.name == "navix":
            return _run_navix_rmax_tabular_training(agent, environment, runner)
        return _run_rmax_tabular_training(agent, environment, runner)

    if policy is None:
        raise ValueError("builtin tabular Q-learning and Sarsa agents require a policy input")
    if replay_buffer.offline_only:
        return _run_offline_tabular_training(agent, policy, environment, runner, replay_buffer)
    if environment.name == "navix":
        return _run_navix_tabular_training(agent, policy, environment, runner, replay_buffer)

    q_table = jnp.full(
        (environment.num_states, environment.num_actions),
        agent.initial_q,
        dtype=jnp.float32,
    )
    action_counts = jnp.zeros_like(q_table)
    buffer_state = initial_replay_buffer(replay_buffer)
    key = jax.random.PRNGKey(runner.seed)
    env_step = make_step_fn(environment)
    train_episodes = _runner_train_episodes(runner)

    def train_episode(carry, episode_idx):
        del episode_idx
        q, counts, buffer_state, scan_key = carry
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            q, counts, buffer_state, key, state, episode_return, episode_length, episode_loss, done = step_carry

            def active_step(active_carry):
                q, counts, buffer_state, key, state, episode_return, episode_length, episode_loss, _ = active_carry
                key, action_key, env_key, update_key, replay_key = jax.random.split(key, 5)
                action = select_action(
                    policy,
                    q[state],
                    counts[state],
                    action_key,
                    training=True,
                    num_actions=environment.num_actions,
                )
                next_state, reward, terminal = env_step(state, action, env_key)
                updated_counts = counts.at[state, action].add(1.0)
                updated_q, td_loss = apply_td_update(
                    agent,
                    policy,
                    q,
                    updated_counts,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                    update_key,
                    num_actions=environment.num_actions,
                )
                updated_buffer = _push_if_enabled(replay_buffer, buffer_state, state, action, reward, next_state, terminal)
                updated_q, replay_loss = _replay_if_ready(
                    agent,
                    policy,
                    replay_buffer,
                    updated_buffer,
                    updated_q,
                    updated_counts,
                    replay_key,
                    environment.num_actions,
                )
                total_loss = _combine_losses(replay_buffer, td_loss, replay_loss)
                return (
                    updated_q,
                    updated_counts,
                    updated_buffer,
                    key,
                    next_state,
                    episode_return + reward,
                    episode_length + 1,
                    episode_loss + total_loss,
                    terminal,
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (q, counts, buffer_state, episode_key, state, episode_return, episode_length, episode_loss, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        q, counts, buffer_state, _, _, episode_return, episode_length, episode_loss, _ = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (q, counts, buffer_state, scan_key), (episode_return, episode_length, mean_loss)

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, state, episode_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, state, episode_return, episode_length, _ = active_carry
                key, action_key, env_key = jax.random.split(key, 3)
                action = select_action(
                    policy,
                    q_final[state],
                    action_counts_final[state],
                    action_key,
                    training=False,
                    num_actions=environment.num_actions,
                )
                next_state, reward, terminal = env_step(state, action, env_key)
                return key, next_state, episode_return + reward, episode_length + 1, terminal

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (episode_key, state, episode_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, episode_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_length)

    @jax.jit
    def run_train_scan(initial_q, initial_counts, initial_buffer, initial_key):
        return jax.lax.scan(
            train_episode,
            (initial_q, initial_counts, initial_buffer, initial_key),
            jnp.arange(train_episodes),
        )

    (q_final, action_counts_final, final_buffer, key), train_history = run_train_scan(
        q_table,
        action_counts,
        buffer_state,
        key,
    )
    if runner.eval_episodes > 0:

        @jax.jit
        def run_eval_scan(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(runner.eval_episodes),
            )

        _, eval_history = run_eval_scan(key)
        eval_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_lengths, train_losses = train_history
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray(eval_returns),
        eval_lengths=np.asarray(eval_lengths),
        dataset=_dataset_from_buffer(final_buffer) if replay_buffer.save_dataset_path else None,
    )


def _run_rmax_tabular_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
) -> TabularRunResult:
    model_state = _initial_rmax_model(agent, environment)
    key = jax.random.PRNGKey(runner.seed)
    env_step = make_step_fn(environment)
    train_episodes = _runner_train_episodes(runner)

    def train_episode(carry, episode_idx):
        del episode_idx
        model, scan_key = carry
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            model, key, state, episode_return, episode_length, episode_loss, done = step_carry

            def active_step(active_carry):
                model, key, state, episode_return, episode_length, episode_loss, _ = active_carry
                key, action_key, env_key = jax.random.split(key, 3)
                action = _rmax_action(model.q_table, state, action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                updated_model, q_delta = _rmax_observe_transition(
                    agent,
                    model,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                )
                return (
                    updated_model,
                    key,
                    next_state,
                    episode_return + reward,
                    episode_length + 1,
                    episode_loss + q_delta,
                    terminal,
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (model, episode_key, state, episode_return, episode_length, episode_loss, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        model, _, _, episode_return, episode_length, episode_loss, _ = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (model, scan_key), (episode_return, episode_length, mean_loss)

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, state, episode_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, state, episode_return, episode_length, _ = active_carry
                key, action_key, env_key = jax.random.split(key, 3)
                action = _rmax_action(q_final, state, action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                return key, next_state, episode_return + reward, episode_length + 1, terminal

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (episode_key, state, episode_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, episode_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_length)

    @jax.jit
    def run_train_scan(initial_model, initial_key):
        return jax.lax.scan(
            train_episode,
            (initial_model, initial_key),
            jnp.arange(train_episodes),
        )

    (final_model, key), train_history = run_train_scan(model_state, key)
    q_final = final_model.q_table
    action_counts_final = final_model.action_counts
    if runner.eval_episodes > 0:

        @jax.jit
        def run_eval_scan(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(runner.eval_episodes),
            )

        _, eval_history = run_eval_scan(key)
        eval_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_lengths, train_losses = train_history
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray(eval_returns),
        eval_lengths=np.asarray(eval_lengths),
    )


def _run_navix_rmax_tabular_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
) -> TabularRunResult:
    from rlflow_builtin.environments.navix import create_navix_environment

    navix_env = create_navix_environment(
        env_name=environment.navix_env_name,
        size=environment.navix_size,
        layout=environment.navix_layout,
        observation_mode=environment.navix_observation_mode,
        action_set=environment.navix_action_set,
        max_steps=environment.navix_max_steps,
    )
    model_state = _initial_rmax_model(agent, environment)
    key = jax.random.PRNGKey(runner.seed)
    train_episodes = _runner_train_episodes(runner)

    def train_episode(carry, episode_idx):
        del episode_idx
        model, scan_key = carry
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            model, key, timestep, state, episode_return, episode_length, episode_loss, done = step_carry

            def active_step(active_carry):
                model, key, timestep, state, episode_return, episode_length, episode_loss, _ = active_carry
                key, action_key = jax.random.split(key)
                action = _rmax_action(model.q_table, state, action_key, environment.num_actions)
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                reward = next_timestep.reward.astype(jnp.float32)
                episode_done = next_timestep.is_done()
                model_terminal = next_timestep.is_termination()
                updated_model, q_delta = _rmax_observe_transition(
                    agent,
                    model,
                    state,
                    action,
                    reward,
                    next_state,
                    model_terminal,
                )
                return (
                    updated_model,
                    key,
                    next_timestep,
                    next_state,
                    episode_return + reward,
                    episode_length + 1,
                    episode_loss + q_delta,
                    episode_done,
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (model, episode_key, timestep, state, episode_return, episode_length, episode_loss, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        model, _, _, _, episode_return, episode_length, episode_loss, _ = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (model, scan_key), (episode_return, episode_length, mean_loss)

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, timestep, state, episode_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, timestep, state, episode_return, episode_length, _ = active_carry
                key, action_key = jax.random.split(key)
                action = _rmax_action(q_final, state, action_key, environment.num_actions)
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                return (
                    key,
                    next_timestep,
                    next_state,
                    episode_return + next_timestep.reward,
                    episode_length + 1,
                    next_timestep.is_done(),
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (episode_key, timestep, state, episode_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, _, episode_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_length)

    @jax.jit
    def run_train_scan(initial_model, initial_key):
        return jax.lax.scan(
            train_episode,
            (initial_model, initial_key),
            jnp.arange(train_episodes),
        )

    (final_model, key), train_history = run_train_scan(model_state, key)
    q_final = final_model.q_table
    action_counts_final = final_model.action_counts
    if runner.eval_episodes > 0:

        @jax.jit
        def run_eval_scan(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(runner.eval_episodes),
            )

        _, eval_history = run_eval_scan(key)
        eval_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_lengths, train_losses = train_history
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray(eval_returns),
        eval_lengths=np.asarray(eval_lengths),
    )


def _initial_rmax_model(agent: AgentConfig, environment: EnvironmentConfig) -> RMaxModelState:
    q_table = jnp.full(
        (environment.num_states, environment.num_actions),
        agent.rmax_v_max,
        dtype=jnp.float32,
    )
    action_counts = jnp.zeros_like(q_table)
    model_counts = jnp.zeros_like(q_table)
    reward_sums = jnp.zeros_like(q_table)
    transition_counts = jnp.zeros(
        (environment.num_states, environment.num_actions, environment.num_states),
        dtype=jnp.float32,
    )
    return RMaxModelState(
        q_table=q_table,
        action_counts=action_counts,
        model_counts=model_counts,
        reward_sums=reward_sums,
        transition_counts=transition_counts,
    )


def _rmax_action(q_table: jax.Array, state: jax.Array, key: jax.Array, num_actions: int) -> jax.Array:
    return greedy(q_table[state], key, num_actions)


def _rmax_observe_transition(
    agent: AgentConfig,
    model: RMaxModelState,
    state: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
) -> tuple[RMaxModelState, jax.Array]:
    was_unknown = model.model_counts[state, action] < float(agent.known_count_threshold)
    model_update = was_unknown.astype(jnp.float32)
    action_counts = model.action_counts.at[state, action].add(1.0)
    model_counts = model.model_counts.at[state, action].add(model_update)
    reward_sums = model.reward_sums.at[state, action].add(reward.astype(jnp.float32) * model_update)
    nonterminal = jnp.logical_not(terminal).astype(jnp.float32)
    transition_counts = model.transition_counts.at[state, action, next_state].add(nonterminal * model_update)
    became_known = jnp.logical_and(
        was_unknown,
        model_counts[state, action] >= float(agent.known_count_threshold),
    )
    q_table = jax.lax.cond(
        became_known,
        lambda _: _rmax_plan_q(
            agent,
            model_counts,
            reward_sums,
            transition_counts,
            model.q_table,
        ),
        lambda _: model.q_table,
        operand=None,
    )
    q_delta = jnp.abs(q_table[state, action] - model.q_table[state, action])
    return (
        RMaxModelState(
            q_table=q_table,
            action_counts=action_counts,
            model_counts=model_counts,
            reward_sums=reward_sums,
            transition_counts=transition_counts,
        ),
        q_delta,
    )


def _rmax_plan_q(
    agent: AgentConfig,
    model_counts: jax.Array,
    reward_sums: jax.Array,
    transition_counts: jax.Array,
    initial_q: jax.Array,
) -> jax.Array:
    known_mask = model_counts >= float(agent.known_count_threshold)
    denom = jnp.maximum(model_counts, 1.0)
    mean_rewards = reward_sums / denom

    def planning_step(q_table, _):
        values = jnp.max(q_table, axis=1)
        expected_next_values = jnp.einsum("san,n->sa", transition_counts, values) / denom
        planned_q = mean_rewards + agent.discount * expected_next_values
        q_table = jnp.where(known_mask, planned_q, agent.rmax_v_max)
        return q_table, None

    q_table, _ = jax.lax.scan(
        planning_step,
        initial_q,
        xs=None,
        length=agent.planning_iterations,
    )
    return q_table


def _run_offline_tabular_training(
    agent: AgentConfig,
    policy: PolicyConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
    replay_buffer: BufferConfig,
) -> TabularRunResult:
    buffer_state = initial_replay_buffer(replay_buffer)
    if int(np.asarray(jax.device_get(buffer_state.size))) <= 0:
        raise ValueError("Offline tabular training requires a non-empty replay dataset")

    q_table = jnp.full(
        (environment.num_states, environment.num_actions),
        agent.initial_q,
        dtype=jnp.float32,
    )
    action_counts = jnp.zeros_like(q_table)
    key = jax.random.PRNGKey(runner.seed)
    train_episodes = _runner_train_episodes(runner)
    total_updates = replay_buffer.offline_updates or runner.train_steps or train_episodes * runner.max_episode_steps
    updates_per_epoch = max(1, int(np.ceil(total_updates / train_episodes)))

    def train_epoch(carry, _):
        q, counts, scan_key = carry
        epoch_loss = jnp.asarray(0.0, dtype=jnp.float32)

        def update_step(step_carry, _):
            q, counts, key, loss_sum = step_carry
            key, sample_key = jax.random.split(key)
            batch = sample_batch(buffer_state, sample_key, replay_buffer.batch_size)

            def update_transition(transition_carry, transition):
                q_batch, counts_batch, transition_key = transition_carry
                transition_key, update_key = jax.random.split(transition_key)
                counts_batch = counts_batch.at[transition.observations, transition.actions].add(1.0)
                q_batch, loss = apply_td_update(
                    agent,
                    policy,
                    q_batch,
                    counts_batch,
                    transition.observations,
                    transition.actions,
                    transition.rewards,
                    transition.next_observations,
                    transition.terminals,
                    update_key,
                    num_actions=environment.num_actions,
                )
                return (q_batch, counts_batch, transition_key), loss

            (q, counts, key), losses = jax.lax.scan(
                update_transition,
                (q, counts, key),
                batch,
            )
            return (q, counts, key, loss_sum + jnp.mean(losses)), None

        q, counts, scan_key, epoch_loss = jax.lax.scan(
            update_step,
            (q, counts, scan_key, epoch_loss),
            xs=None,
            length=updates_per_epoch,
        )[0]
        return (q, counts, scan_key), (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(updates_per_epoch, dtype=jnp.int32),
            epoch_loss / updates_per_epoch,
        )

    @jax.jit
    def run_train_scan(initial_q, initial_counts, initial_key):
        return jax.lax.scan(
            train_epoch,
            (initial_q, initial_counts, initial_key),
            xs=None,
            length=train_episodes,
        )

    (q_final, action_counts_final, _), train_history = run_train_scan(q_table, action_counts, key)
    train_returns, train_lengths, train_losses = train_history
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray([], dtype=np.float32),
        eval_lengths=np.asarray([], dtype=np.int32),
        dataset=_dataset_from_buffer(buffer_state) if replay_buffer.save_dataset_path else None,
    )


def _run_navix_tabular_training(
    agent: AgentConfig,
    policy: PolicyConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
    replay_buffer: BufferConfig | None = None,
) -> TabularRunResult:
    from rlflow_builtin.environments.navix import create_navix_environment

    replay_buffer = replay_buffer or no_buffer_config()
    navix_env = create_navix_environment(
        env_name=environment.navix_env_name,
        size=environment.navix_size,
        layout=environment.navix_layout,
        observation_mode=environment.navix_observation_mode,
        action_set=environment.navix_action_set,
        max_steps=environment.navix_max_steps,
    )
    q_table = jnp.full(
        (environment.num_states, environment.num_actions),
        agent.initial_q,
        dtype=jnp.float32,
    )
    action_counts = jnp.zeros_like(q_table)
    buffer_state = initial_replay_buffer(replay_buffer)
    key = jax.random.PRNGKey(runner.seed)
    train_episodes = _runner_train_episodes(runner)

    def train_episode(carry, episode_idx):
        del episode_idx
        q, counts, buffer_state, scan_key = carry
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            q, counts, buffer_state, key, timestep, state, episode_return, episode_length, episode_loss, done = step_carry

            def active_step(active_carry):
                q, counts, buffer_state, key, timestep, state, episode_return, episode_length, episode_loss, _ = active_carry
                key, action_key, update_key, replay_key = jax.random.split(key, 4)
                action = select_action(
                    policy,
                    q[state],
                    counts[state],
                    action_key,
                    training=True,
                    num_actions=environment.num_actions,
                )
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                reward = next_timestep.reward.astype(jnp.float32)
                terminal = next_timestep.is_done()
                updated_counts = counts.at[state, action].add(1.0)
                updated_q, td_loss = apply_td_update(
                    agent,
                    policy,
                    q,
                    updated_counts,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                    update_key,
                    num_actions=environment.num_actions,
                )
                updated_buffer = _push_if_enabled(
                    replay_buffer,
                    buffer_state,
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                )
                updated_q, replay_loss = _replay_if_ready(
                    agent,
                    policy,
                    replay_buffer,
                    updated_buffer,
                    updated_q,
                    updated_counts,
                    replay_key,
                    environment.num_actions,
                )
                total_loss = _combine_losses(replay_buffer, td_loss, replay_loss)
                return (
                    updated_q,
                    updated_counts,
                    updated_buffer,
                    key,
                    next_timestep,
                    next_state,
                    episode_return + reward,
                    episode_length + 1,
                    episode_loss + total_loss,
                    terminal,
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (q, counts, buffer_state, episode_key, timestep, state, episode_return, episode_length, episode_loss, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        q, counts, buffer_state, _, _, _, episode_return, episode_length, episode_loss, _ = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (q, counts, buffer_state, scan_key), (episode_return, episode_length, mean_loss)

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, timestep, state, episode_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, timestep, state, episode_return, episode_length, _ = active_carry
                key, action_key = jax.random.split(key)
                action = select_action(
                    policy,
                    q_final[state],
                    action_counts_final[state],
                    action_key,
                    training=False,
                    num_actions=environment.num_actions,
                )
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                return (
                    key,
                    next_timestep,
                    next_state,
                    episode_return + next_timestep.reward,
                    episode_length + 1,
                    next_timestep.is_done(),
                )

            return jax.lax.cond(
                done,
                lambda inactive_carry: inactive_carry,
                active_step,
                step_carry,
            ), None

        carry_out, _ = jax.lax.scan(
            step_fn,
            (episode_key, timestep, state, episode_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, _, episode_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_length)

    @jax.jit
    def run_train_scan(initial_q, initial_counts, initial_buffer, initial_key):
        return jax.lax.scan(
            train_episode,
            (initial_q, initial_counts, initial_buffer, initial_key),
            jnp.arange(train_episodes),
        )

    (q_final, action_counts_final, final_buffer, key), train_history = run_train_scan(
        q_table,
        action_counts,
        buffer_state,
        key,
    )
    if runner.eval_episodes > 0:

        @jax.jit
        def run_eval_scan(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(runner.eval_episodes),
            )

        _, eval_history = run_eval_scan(key)
        eval_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_lengths, train_losses = train_history
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray(eval_returns),
        eval_lengths=np.asarray(eval_lengths),
        dataset=_dataset_from_buffer(final_buffer) if replay_buffer.save_dataset_path else None,
    )


def _dataset_from_buffer(buffer_state: ReplayBufferState) -> TabularDataset:
    arrays = replay_dataset_arrays(buffer_state)
    return TabularDataset(
        observations=arrays["observations"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        next_observations=arrays["next_observations"],
        terminals=arrays["terminals"],
    )


def _runner_train_episodes(runner: RunnerConfig) -> int:
    if runner.train_steps is None:
        return runner.train_episodes
    return max(1, int(np.ceil(runner.train_steps / runner.max_episode_steps)))


def _push_if_enabled(
    replay_buffer: BufferConfig,
    buffer_state: ReplayBufferState,
    state: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
) -> ReplayBufferState:
    if not replay_buffer.enabled:
        return buffer_state
    return push_transition(buffer_state, state, action, reward, next_state, terminal)


def _replay_if_ready(
    agent: AgentConfig,
    policy: PolicyConfig,
    replay_buffer: BufferConfig,
    buffer_state: ReplayBufferState,
    q_table: jax.Array,
    action_counts: jax.Array,
    key: jax.Array,
    num_actions: int,
) -> tuple[jax.Array, jax.Array]:
    if not replay_buffer.enabled or replay_buffer.updates_per_step <= 0:
        return q_table, jnp.asarray(0.0, dtype=jnp.float32)

    def replay_updates(args):
        q_in, replay_key = args

        def update_batch(carry, _):
            q_batch, batch_key = carry
            batch_key, sample_key, td_key = jax.random.split(batch_key, 3)
            batch = sample_batch(buffer_state, sample_key, replay_buffer.batch_size)
            (q_batch, _), losses = jax.lax.scan(
                lambda update_carry, transition: _update_replay_transition(
                    agent,
                    policy,
                    action_counts,
                    update_carry,
                    transition,
                    num_actions,
                ),
                (q_batch, td_key),
                batch,
            )
            return (q_batch, batch_key), jnp.mean(losses)

        (q_out, _), replay_losses = jax.lax.scan(
            update_batch,
            (q_in, replay_key),
            xs=None,
            length=replay_buffer.updates_per_step,
        )
        return q_out, jnp.mean(replay_losses)

    ready = buffer_state.size >= replay_buffer.min_size
    return jax.lax.cond(
        ready,
        replay_updates,
        lambda args: (args[0], jnp.asarray(0.0, dtype=jnp.float32)),
        (q_table, key),
    )


def _update_replay_transition(
    agent: AgentConfig,
    policy: PolicyConfig,
    action_counts: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    transition: TransitionBatch,
    num_actions: int,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    q_table, key = carry
    key, update_key = jax.random.split(key)
    updated_q, td_loss = apply_td_update(
        agent,
        policy,
        q_table,
        action_counts,
        transition.observations,
        transition.actions,
        transition.rewards,
        transition.next_observations,
        transition.terminals,
        update_key,
        num_actions=num_actions,
    )
    return (updated_q, key), td_loss


def _combine_losses(replay_buffer: BufferConfig, td_loss: jax.Array, replay_loss: jax.Array) -> jax.Array:
    if not replay_buffer.enabled or replay_buffer.updates_per_step <= 0:
        return td_loss
    denominator = jnp.where(replay_loss > 0.0, 2.0, 1.0)
    return (td_loss + replay_loss) / denominator
