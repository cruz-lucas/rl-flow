from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.tabular.agents import apply_td_update
from rlflow_builtin.tabular.buffers import (
    initial_replay_buffer,
    no_buffer_config,
    push_transition,
    replay_dataset_arrays,
    sample_batch,
)
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


def _select_next_action(
    agent: AgentConfig,
    policy: PolicyConfig,
    q_table: jax.Array,
    action_counts: jax.Array,
    next_state: jax.Array,
    key: jax.Array,
    num_actions: int,
) -> jax.Array:
    """Select an on-policy action at ``next_state`` (used for SARSA bootstrapping
    and for choosing the next executed action in the online loops)."""
    del agent
    return select_action(
        policy,
        q_table[next_state],
        action_counts[next_state],
        key,
        training=True,
        num_actions=num_actions,
    )


def _run_eval_scan(evaluate_episode, key, eval_episodes: int):
    """Run ``eval_episodes`` greedy rollouts (or return empty arrays when zero).

    Shared by every rollout-based tabular path; ``evaluate_episode`` is the
    path-specific per-episode closure, so the scan/RNG behaviour is unchanged.
    """
    if eval_episodes > 0:

        @jax.jit
        def run_eval(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(eval_episodes),
            )

        _, eval_history = run_eval(key)
        return eval_history
    return (
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.int32),
    )


def _tabular_result(
    q_final,
    action_counts_final,
    train_history,
    eval_arrays,
    *,
    dataset: TabularDataset | None = None,
) -> TabularRunResult:
    """Assemble a :class:`TabularRunResult` from the train/eval scan outputs."""
    train_returns, train_discounted_returns, train_lengths, train_losses = train_history
    eval_returns, eval_discounted_returns, eval_lengths = eval_arrays
    return TabularRunResult(
        q_table=np.asarray(q_final),
        action_counts=np.asarray(action_counts_final),
        train_returns=np.asarray(train_returns),
        train_discounted_returns=np.asarray(train_discounted_returns),
        train_lengths=np.asarray(train_lengths),
        train_losses=np.asarray(train_losses),
        eval_returns=np.asarray(eval_returns),
        eval_discounted_returns=np.asarray(eval_discounted_returns),
        eval_lengths=np.asarray(eval_lengths),
        dataset=dataset,
    )


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

    if agent.algorithm == "mbie_eb":
        if replay_buffer.enabled:
            raise ValueError("builtin.agent.mbie_eb_tabular does not support replay buffers")
        if environment.name == "navix":
            raise ValueError("builtin.agent.mbie_eb_tabular does not support navix environments")
        return _run_mbie_tabular_training(agent, environment, runner)

    if agent.algorithm in ("replay_rmax", "replay_mbie_eb"):
        if not replay_buffer.enabled:
            raise ValueError(
                "builtin.agent.replay_rmax_tabular / replay_mbie_eb_tabular require a "
                "builtin.replay.tabular_uniform input"
            )
        if environment.name == "navix":
            raise ValueError(
                "replay-based optimistic tabular agents do not support navix environments"
            )
        return _run_replay_optimistic_training(agent, environment, runner, replay_buffer)

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
        scan_key, episode_key, reset_key, first_action_key = jax.random.split(scan_key, 4)
        state = initial_state(environment, reset_key)
        action = select_action(
            policy,
            q[state],
            counts[state],
            first_action_key,
            training=True,
            num_actions=environment.num_actions,
        )
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                q,
                counts,
                buffer_state,
                key,
                state,
                action,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    q,
                    counts,
                    buffer_state,
                    key,
                    state,
                    action,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    episode_loss,
                    _,
                ) = active_carry
                key, env_key, next_action_key, replay_key = jax.random.split(key, 4)
                next_state, reward, terminal = env_step(state, action, env_key)
                updated_counts = counts.at[state, action].add(1.0)
                next_action = _select_next_action(
                    agent,
                    policy,
                    q,
                    updated_counts,
                    next_state,
                    next_action_key,
                    environment.num_actions,
                )
                update_reward = _apply_count_bonus(agent, reward, updated_counts, state, action)
                updated_q, td_loss = apply_td_update(
                    agent,
                    policy,
                    q,
                    updated_counts,
                    state,
                    action,
                    update_reward,
                    next_state,
                    terminal,
                    next_action_key,
                    num_actions=environment.num_actions,
                    next_action=next_action,
                )
                executed_next_action = (
                    next_action
                    if agent.algorithm == "sarsa"
                    else _select_next_action(
                        agent,
                        policy,
                        updated_q,
                        updated_counts,
                        next_state,
                        next_action_key,
                        environment.num_actions,
                    )
                )
                updated_buffer = _push_if_enabled(
                    replay_buffer, buffer_state, state, action, reward, next_state, terminal
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
                    next_state,
                    executed_next_action,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                q,
                counts,
                buffer_state,
                episode_key,
                state,
                action,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        (
            q,
            counts,
            buffer_state,
            _,
            _,
            _,
            episode_return,
            episode_discounted_return,
            episode_length,
            episode_loss,
            _,
        ) = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (q, counts, buffer_state, scan_key), (
            episode_return,
            episode_discounted_return,
            episode_length,
            mean_loss,
        )

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, state, episode_return, episode_discounted_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, state, episode_return, episode_discounted_return, episode_length, _ = (
                    active_carry
                )
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
                return (
                    key,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
                    episode_length + 1,
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
            (episode_key, state, episode_return, episode_discounted_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_discounted_return, episode_length)

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
    eval_arrays = _run_eval_scan(evaluate_episode, key, runner.eval_episodes)
    return _tabular_result(
        q_final,
        action_counts_final,
        train_history,
        eval_arrays,
        dataset=_dataset_from_buffer(final_buffer) if replay_buffer.save_dataset_path else None,
    )


def _run_model_based_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
    observe_fn: Callable[..., tuple[RMaxModelState, jax.Array]],
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
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                model,
                key,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    model,
                    key,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    episode_loss,
                    _,
                ) = active_carry
                key, action_key, env_key = jax.random.split(key, 3)
                action = _rmax_action(model.q_table, state, action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                updated_model, q_delta = observe_fn(
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
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                model,
                episode_key,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        model, _, _, episode_return, episode_discounted_return, episode_length, episode_loss, _ = (
            carry_out
        )
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (model, scan_key), (
            episode_return,
            episode_discounted_return,
            episode_length,
            mean_loss,
        )

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, state, episode_return, episode_discounted_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, state, episode_return, episode_discounted_return, episode_length, _ = (
                    active_carry
                )
                key, action_key, env_key = jax.random.split(key, 3)
                action = _rmax_action(q_final, state, action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                return (
                    key,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
                    episode_length + 1,
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
            (episode_key, state, episode_return, episode_discounted_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_discounted_return, episode_length)

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
    eval_arrays = _run_eval_scan(evaluate_episode, key, runner.eval_episodes)
    return _tabular_result(q_final, action_counts_final, train_history, eval_arrays)


def _run_rmax_tabular_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
) -> TabularRunResult:
    return _run_model_based_training(agent, environment, runner, _rmax_observe_transition)


def _run_mbie_tabular_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
) -> TabularRunResult:
    return _run_model_based_training(agent, environment, runner, _mbie_observe_transition)


def _mbie_observe_transition(
    agent: AgentConfig,
    model: RMaxModelState,
    state: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
) -> tuple[RMaxModelState, jax.Array]:
    """MBIE-EB model update: every visit accumulates the empirical model and the
    optimistic value function is re-planned (the ``beta/sqrt(N)`` bonus shrinks with
    each visit, so unlike R-Max we re-plan on every step rather than at a threshold)."""
    action_counts = model.action_counts.at[state, action].add(1.0)
    model_counts = model.model_counts.at[state, action].add(1.0)
    reward_sums = model.reward_sums.at[state, action].add(reward.astype(jnp.float32))
    nonterminal = jnp.logical_not(terminal).astype(jnp.float32)
    transition_counts = model.transition_counts.at[state, action, next_state].add(nonterminal)
    q_table = _mbie_plan_q(agent, model_counts, reward_sums, transition_counts, model.q_table)
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


def _mbie_plan_q(
    agent: AgentConfig,
    model_counts: jax.Array,
    reward_sums: jax.Array,
    transition_counts: jax.Array,
    initial_q: jax.Array,
) -> jax.Array:
    """Value iteration for MBIE-EB: visited (s,a) use the empirical model plus a
    ``beta/sqrt(N(s,a))`` exploration bonus; unvisited (s,a) stay optimistic (V_max)."""
    known_mask = model_counts > 0.0
    denom = jnp.maximum(model_counts, 1.0)
    mean_rewards = reward_sums / denom
    bonus = jnp.where(known_mask, agent.mbie_beta / jnp.sqrt(denom), 0.0)

    def planning_step(q_table, _):
        values = jnp.max(q_table, axis=1)
        expected_next_values = jnp.einsum("san,n->sa", transition_counts, values) / denom
        planned_q = mean_rewards + bonus + agent.discount * expected_next_values
        q_table = jnp.where(known_mask, planned_q, agent.rmax_v_max)
        return q_table, None

    q_table, _ = jax.lax.scan(
        planning_step,
        initial_q,
        xs=None,
        length=agent.planning_iterations,
    )
    return q_table


def _known_mask(agent: AgentConfig, counts: jax.Array) -> jax.Array:
    """Optimism predicate for the replay-based agents (Algorithm 1 rule ``C``).

    ``replay_rmax``  -> known when ``N(s,a) >= m`` (``known_count_threshold``);
    ``replay_mbie_eb`` -> known when ``N(s,a) > 0``. ``agent.algorithm`` is a static
    Python string, so the branch is resolved at trace time."""
    if agent.algorithm == "replay_rmax":
        return counts >= float(agent.known_count_threshold)
    return counts > 0.0


def _optimistic_values(agent: AgentConfig, q_table: jax.Array, counts: jax.Array) -> jax.Array:
    """U-table (Algorithm 1 rule ``U``): ``Q`` where known, ``V_max`` otherwise."""
    return jnp.where(_known_mask(agent, counts), q_table, agent.rmax_v_max)


def _optimistic_bonus(
    agent: AgentConfig, counts: jax.Array, state: jax.Array, action: jax.Array
) -> jax.Array:
    """Target bonus ``beta/sqrt(max(N(s,a),1))`` (MBIE variant; 0 for the R-Max variant
    since ``mbie_beta`` is 0.0 there)."""
    if agent.mbie_beta == 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    visit = jnp.maximum(counts[state, action], 1.0)
    return agent.mbie_beta / jnp.sqrt(visit)


def _update_optimistic_transition(
    agent: AgentConfig,
    counts: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    transition: TransitionBatch,
    num_actions: int,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Predicate-gated TD update with optimistic bootstrap (Algorithm 1 lines 13-16).

    ``counts`` is held fixed across a replay pass; only known ``(s,a)`` are updated.
    The bootstrap uses ``max_{a'} U(s',a')`` so unknown successors contribute ``V_max``."""
    del num_actions
    q_table, key = carry
    state = transition.observations
    action = transition.actions
    reward = transition.rewards
    next_state = transition.next_observations
    terminal = transition.terminals.astype(jnp.float32)

    next_u_row = jnp.where(
        _known_mask(agent, counts[next_state]), q_table[next_state], agent.rmax_v_max
    )
    next_value = jnp.max(next_u_row)
    bonus = _optimistic_bonus(agent, counts, state, action)
    target = reward + bonus + agent.discount * next_value * (1.0 - terminal)

    q_old = q_table[state, action]
    predicate = _known_mask(agent, counts[state, action])
    new_value = q_old + agent.learning_rate * (target - q_old)
    q_table = q_table.at[state, action].set(jnp.where(predicate, new_value, q_old))
    loss = jnp.where(predicate, jnp.abs(target - q_old), 0.0)
    return (q_table, key), loss


def _replay_optimistic(
    agent: AgentConfig,
    replay_buffer: BufferConfig,
    buffer_state: ReplayBufferState,
    q_table: jax.Array,
    counts: jax.Array,
    key: jax.Array,
    num_actions: int,
) -> tuple[jax.Array, jax.Array]:
    """Replay the buffer for one env step: until convergence (max|ΔQ| < tol) when
    ``replay_until_convergence``, else ``max(updates_per_step, 1)`` minibatch passes."""
    batch_size = replay_buffer.batch_size

    def one_batch(q_in: jax.Array, batch_key: jax.Array):
        batch_key, sample_key, update_key = jax.random.split(batch_key, 3)
        batch = sample_batch(buffer_state, sample_key, batch_size)
        (q_out, _), losses = jax.lax.scan(
            lambda carry, transition: _update_optimistic_transition(
                agent, counts, carry, transition, num_actions
            ),
            (q_in, update_key),
            batch,
        )
        return q_out, batch_key, jnp.mean(losses)

    def do_replay(args):
        q_in, replay_key = args
        if replay_buffer.replay_until_convergence:
            tol = replay_buffer.convergence_tol
            max_iters = replay_buffer.max_replay_iters

            def cond(state):
                _, _, iters, delta = state
                return jnp.logical_and(iters < max_iters, delta > tol)

            def body(state):
                q_cur, batch_key, iters, _ = state
                q_next, batch_key, _ = one_batch(q_cur, batch_key)
                delta = jnp.max(jnp.abs(q_next - q_cur))
                return (q_next, batch_key, iters + 1, delta)

            init = (
                q_in,
                replay_key,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            q_out, _, _, delta_out = jax.lax.while_loop(cond, body, init)
            return q_out, delta_out

        def scan_body(carry, _):
            q_cur, batch_key = carry
            q_next, batch_key, loss = one_batch(q_cur, batch_key)
            return (q_next, batch_key), loss

        (q_out, _), losses = jax.lax.scan(
            scan_body,
            (q_in, replay_key),
            xs=None,
            length=max(replay_buffer.updates_per_step, 1),
        )
        return q_out, jnp.mean(losses)

    ready = buffer_state.size >= replay_buffer.min_size
    return jax.lax.cond(
        ready,
        do_replay,
        lambda args: (args[0], jnp.asarray(0.0, dtype=jnp.float32)),
        (q_table, key),
    )


def _run_replay_optimistic_training(
    agent: AgentConfig,
    environment: EnvironmentConfig,
    runner: RunnerConfig,
    replay_buffer: BufferConfig,
) -> TabularRunResult:
    """Algorithm 1 (Replay-Based Optimistic Q-Learning). ``Q`` starts at 0; optimism is
    injected through ``U(s,a)=V_max`` for unknown ``(s,a)``. Each env step acts greedily
    on ``U``, stores the transition, then replays the buffer, updating only known
    ``(s,a)`` toward optimistic bootstrap targets."""
    q_table = jnp.zeros((environment.num_states, environment.num_actions), dtype=jnp.float32)
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
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                q,
                counts,
                buffer_state,
                key,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    q,
                    counts,
                    buffer_state,
                    key,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    episode_loss,
                    _,
                ) = active_carry
                key, action_key, env_key, replay_key = jax.random.split(key, 4)
                u_table = _optimistic_values(agent, q, counts)
                action = greedy(u_table[state], action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                counts = counts.at[state, action].add(1.0)
                buffer_state = push_transition(
                    buffer_state, state, action, reward, next_state, terminal
                )
                q, replay_loss = _replay_optimistic(
                    agent,
                    replay_buffer,
                    buffer_state,
                    q,
                    counts,
                    replay_key,
                    environment.num_actions,
                )
                return (
                    q,
                    counts,
                    buffer_state,
                    key,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
                    episode_length + 1,
                    episode_loss + replay_loss,
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
            (
                q,
                counts,
                buffer_state,
                episode_key,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        (
            q,
            counts,
            buffer_state,
            _,
            _,
            episode_return,
            episode_discounted_return,
            episode_length,
            episode_loss,
            _,
        ) = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (q, counts, buffer_state, scan_key), (
            episode_return,
            episode_discounted_return,
            episode_length,
            mean_loss,
        )

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        state = initial_state(environment, reset_key)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            key, state, episode_return, episode_discounted_return, episode_length, done = step_carry

            def active_step(active_carry):
                key, state, episode_return, episode_discounted_return, episode_length, _ = (
                    active_carry
                )
                key, action_key, env_key = jax.random.split(key, 3)
                u_row = jnp.where(
                    _known_mask(agent, action_counts_final[state]),
                    q_final[state],
                    agent.rmax_v_max,
                )
                action = greedy(u_row, action_key, environment.num_actions)
                next_state, reward, terminal = env_step(state, action, env_key)
                return (
                    key,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
                    episode_length + 1,
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
            (episode_key, state, episode_return, episode_discounted_return, episode_length, done),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_discounted_return, episode_length)

    @jax.jit
    def run_train_scan(initial_q, initial_counts, initial_buffer, initial_key):
        return jax.lax.scan(
            train_episode,
            (initial_q, initial_counts, initial_buffer, initial_key),
            jnp.arange(train_episodes),
        )

    (q_final, action_counts_final, _, key), train_history = run_train_scan(
        q_table,
        action_counts,
        buffer_state,
        key,
    )
    eval_arrays = _run_eval_scan(evaluate_episode, key, runner.eval_episodes)
    return _tabular_result(q_final, action_counts_final, train_history, eval_arrays)


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
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                model,
                key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    model,
                    key,
                    timestep,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    episode_loss,
                    _,
                ) = active_carry
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
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                model,
                episode_key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        (
            model,
            _,
            _,
            _,
            episode_return,
            episode_discounted_return,
            episode_length,
            episode_loss,
            _,
        ) = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (model, scan_key), (
            episode_return,
            episode_discounted_return,
            episode_length,
            mean_loss,
        )

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    key,
                    timestep,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    _,
                ) = active_carry
                key, action_key = jax.random.split(key)
                action = _rmax_action(q_final, state, action_key, environment.num_actions)
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                reward = next_timestep.reward.astype(jnp.float32)
                return (
                    key,
                    next_timestep,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                episode_key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_discounted_return, episode_length)

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
    eval_arrays = _run_eval_scan(evaluate_episode, key, runner.eval_episodes)
    return _tabular_result(q_final, action_counts_final, train_history, eval_arrays)


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


def _rmax_action(
    q_table: jax.Array, state: jax.Array, key: jax.Array, num_actions: int
) -> jax.Array:
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
    transition_counts = model.transition_counts.at[state, action, next_state].add(
        nonterminal * model_update
    )
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
    total_updates = (
        replay_buffer.offline_updates
        or runner.train_steps
        or train_episodes * runner.max_episode_steps
    )
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

    (q_final, action_counts_final, eval_key), train_history = run_train_scan(
        q_table, action_counts, key
    )

    if runner.eval_episodes > 0 and environment.name != "navix":
        env_step = make_step_fn(environment)

        def evaluate_episode(scan_key):
            scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
            state = initial_state(environment, reset_key)
            episode_return = jnp.asarray(0.0, dtype=jnp.float32)
            episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
            episode_length = jnp.asarray(0, dtype=jnp.int32)
            done = jnp.asarray(False)

            def step_fn(step_carry, _):
                key, state, episode_return, episode_discounted_return, episode_length, done = (
                    step_carry
                )

                def active_step(active_carry):
                    key, state, episode_return, episode_discounted_return, episode_length, _ = (
                        active_carry
                    )
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
                    return (
                        key,
                        next_state,
                        episode_return + reward,
                        episode_discounted_return + (agent.discount**episode_length) * reward,
                        episode_length + 1,
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
                (
                    episode_key,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    done,
                ),
                xs=None,
                length=runner.max_episode_steps,
            )
            _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
            return scan_key, (episode_return, episode_discounted_return, episode_length)

        @jax.jit
        def run_eval_scan(initial_key):
            return jax.lax.scan(
                lambda carry, _: evaluate_episode(carry),
                initial_key,
                jnp.arange(runner.eval_episodes),
            )

        _, eval_history = run_eval_scan(eval_key)
        eval_returns, eval_discounted_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_discounted_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    return _tabular_result(
        q_final,
        action_counts_final,
        train_history,
        (eval_returns, eval_discounted_returns, eval_lengths),
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
        scan_key, episode_key, reset_key, first_action_key = jax.random.split(scan_key, 4)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        # Fix #4: carry the acting action so SARSA bootstraps with the actually
        # taken next action (see the non-navix online loop for details).
        action = select_action(
            policy,
            q[state],
            counts[state],
            first_action_key,
            training=True,
            num_actions=environment.num_actions,
        )
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_loss = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                q,
                counts,
                buffer_state,
                key,
                timestep,
                state,
                action,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    q,
                    counts,
                    buffer_state,
                    key,
                    timestep,
                    state,
                    action,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    episode_loss,
                    _,
                ) = active_carry
                key, next_action_key, replay_key = jax.random.split(key, 3)
                next_timestep = navix_env.step(timestep, action)
                next_state = next_timestep.observation.astype(jnp.int32)
                reward = next_timestep.reward.astype(jnp.float32)
                terminal = next_timestep.is_done()
                updated_counts = counts.at[state, action].add(1.0)
                next_action = _select_next_action(
                    agent,
                    policy,
                    q,
                    updated_counts,
                    next_state,
                    next_action_key,
                    environment.num_actions,
                )
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
                    next_action_key,
                    num_actions=environment.num_actions,
                    next_action=next_action,
                )
                executed_next_action = (
                    next_action
                    if agent.algorithm == "sarsa"
                    else _select_next_action(
                        agent,
                        policy,
                        updated_q,
                        updated_counts,
                        next_state,
                        next_action_key,
                        environment.num_actions,
                    )
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
                    executed_next_action,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                q,
                counts,
                buffer_state,
                episode_key,
                timestep,
                state,
                action,
                episode_return,
                episode_discounted_return,
                episode_length,
                episode_loss,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        (
            q,
            counts,
            buffer_state,
            _,
            _,
            _,
            _,
            episode_return,
            episode_discounted_return,
            episode_length,
            episode_loss,
            _,
        ) = carry_out
        mean_loss = episode_loss / jnp.maximum(episode_length, 1)
        return (q, counts, buffer_state, scan_key), (
            episode_return,
            episode_discounted_return,
            episode_length,
            mean_loss,
        )

    def evaluate_episode(scan_key):
        scan_key, episode_key, reset_key = jax.random.split(scan_key, 3)
        timestep = navix_env.reset(reset_key)
        state = timestep.observation.astype(jnp.int32)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_discounted_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        done = jnp.asarray(False)

        def step_fn(step_carry, _):
            (
                key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                done,
            ) = step_carry

            def active_step(active_carry):
                (
                    key,
                    timestep,
                    state,
                    episode_return,
                    episode_discounted_return,
                    episode_length,
                    _,
                ) = active_carry
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
                reward = next_timestep.reward.astype(jnp.float32)
                return (
                    key,
                    next_timestep,
                    next_state,
                    episode_return + reward,
                    episode_discounted_return + (agent.discount**episode_length) * reward,
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
            (
                episode_key,
                timestep,
                state,
                episode_return,
                episode_discounted_return,
                episode_length,
                done,
            ),
            xs=None,
            length=runner.max_episode_steps,
        )
        _, _, _, episode_return, episode_discounted_return, episode_length, _ = carry_out
        return scan_key, (episode_return, episode_discounted_return, episode_length)

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
    eval_arrays = _run_eval_scan(evaluate_episode, key, runner.eval_episodes)
    return _tabular_result(
        q_final,
        action_counts_final,
        train_history,
        eval_arrays,
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


def _apply_count_bonus(
    agent: AgentConfig,
    reward: jax.Array,
    counts: jax.Array,
    state: jax.Array,
    action: jax.Array,
) -> jax.Array:
    """Add the count-based intrinsic bonus ``beta / sqrt(max(N(s,a), 1))`` to a reward.

    ``count_bonus_beta`` is a static Python float, so at 0.0 this returns ``reward``
    unchanged (identical graph) — plain Q-learning is preserved exactly.
    """
    if agent.count_bonus_beta == 0.0:
        return reward
    visit = jnp.maximum(counts[state, action], 1.0)
    return reward + agent.count_bonus_beta / jnp.sqrt(visit)


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
    if not replay_buffer.enabled:
        return q_table, jnp.asarray(0.0, dtype=jnp.float32)

    ready = buffer_state.size >= replay_buffer.min_size

    def _replay_batch(carry, _):
        q_batch, batch_key = carry
        batch_key, sample_key, td_key = jax.random.split(batch_key, 3)
        batch = sample_batch(buffer_state, sample_key, replay_buffer.batch_size)
        (q_next, _), losses = jax.lax.scan(
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
        return (q_next, batch_key), jnp.mean(losses)

    if replay_buffer.replay_until_convergence:
        tol = replay_buffer.convergence_tol
        max_iters = replay_buffer.max_replay_iters

        def replay_until_converged(args):
            q_in, replay_key = args

            def cond(state):
                _, _, iters, delta = state
                return jnp.logical_and(iters < max_iters, delta > tol)

            def body(state):
                q_cur, batch_key, iters, _ = state
                (q_next, batch_key), _ = _replay_batch((q_cur, batch_key), None)
                delta = jnp.max(jnp.abs(q_next - q_cur))
                return (q_next, batch_key, iters + 1, delta)

            init = (
                q_in,
                replay_key,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            q_out, _, _, delta_out = jax.lax.while_loop(cond, body, init)
            return q_out, delta_out

        return jax.lax.cond(
            ready,
            replay_until_converged,
            lambda args: (args[0], jnp.asarray(0.0, dtype=jnp.float32)),
            (q_table, key),
        )

    if replay_buffer.updates_per_step <= 0:
        return q_table, jnp.asarray(0.0, dtype=jnp.float32)

    def replay_updates(args):
        q_in, replay_key = args
        (q_out, _), replay_losses = jax.lax.scan(
            _replay_batch,
            (q_in, replay_key),
            xs=None,
            length=replay_buffer.updates_per_step,
        )
        return q_out, jnp.mean(replay_losses)

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
    update_reward = _apply_count_bonus(
        agent,
        transition.rewards,
        action_counts,
        transition.observations,
        transition.actions,
    )
    updated_q, td_loss = apply_td_update(
        agent,
        policy,
        q_table,
        action_counts,
        transition.observations,
        transition.actions,
        update_reward,
        transition.next_observations,
        transition.terminals,
        update_key,
        num_actions=num_actions,
    )
    return (updated_q, key), td_loss


def _combine_losses(
    replay_buffer: BufferConfig, td_loss: jax.Array, replay_loss: jax.Array
) -> jax.Array:
    replay_active = replay_buffer.enabled and (
        replay_buffer.replay_until_convergence or replay_buffer.updates_per_step > 0
    )
    if not replay_active:
        return td_loss
    denominator = jnp.where(replay_loss > 0.0, 2.0, 1.0)
    return (td_loss + replay_loss) / denominator
