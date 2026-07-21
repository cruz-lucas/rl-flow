"""Regression tests for tabular training correctness fixes.

Covers:
  * discounted returns are computed and surfaced by every tabular path;
  * offline training reports real greedy-evaluation returns instead of zeros;
  * on-policy SARSA runs with the carried-action bootstrap;
  * RiverSwim honours non-positive easy/hard reward configs;
  * the DQN epsilon schedule tolerates ``epsilon_decay_steps == 0``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.dqn.training import _epsilon
from rlflow_builtin.tabular.environments import make_step_fn, riverswim_step
from rlflow_builtin.tabular.training import run_tabular_training
from rlflow_builtin.tabular.types import (
    AgentConfig,
    BufferConfig,
    EnvironmentConfig,
    PolicyConfig,
    RunnerConfig,
)


def _runner(train_episodes: int = 8, eval_episodes: int = 4, max_steps: int = 25) -> RunnerConfig:
    return RunnerConfig(
        seed=0,
        train_episodes=train_episodes,
        train_steps=None,
        max_episode_steps=max_steps,
        eval_episodes=eval_episodes,
        checkpoint_freq=None,
        checkpoint_dir="checkpoints",
        save_final_checkpoint=False,
    )


_GRID = EnvironmentConfig(
    name="gridworld",
    num_states=9,
    num_actions=4,
    start_state=0,
    width=3,
    height=3,
    goal_state=8,
    goal_reward=1.0,
    step_reward=0.0,
)
_POLICY = PolicyConfig(name="epsilon_greedy", epsilon=0.2)


def test_online_run_reports_discounted_returns_that_differ_from_undiscounted() -> None:
    agent = AgentConfig(algorithm="q_learning", discount=0.9, learning_rate=0.3, initial_q=0.0)
    result = run_tabular_training(agent, _POLICY, _GRID, _runner())

    assert result.train_discounted_returns.shape == result.train_returns.shape
    assert result.eval_discounted_returns.shape == result.eval_returns.shape

    # Where a positive (undiscounted) return was earned, discounting with
    # gamma < 1 must strictly reduce it (the goal reward arrives after >0 steps).
    earned = result.eval_returns > 0
    assert earned.any(), "expected the agent to reach the goal at least once during eval"
    assert np.all(result.eval_discounted_returns[earned] < result.eval_returns[earned])


def test_sarsa_runs_and_learns() -> None:
    agent = AgentConfig(algorithm="sarsa", discount=0.9, learning_rate=0.3, initial_q=0.0)
    result = run_tabular_training(agent, _POLICY, _GRID, _runner(train_episodes=40))
    assert result.train_discounted_returns.shape == result.train_returns.shape
    # A greedy SARSA policy should solve the tiny grid at least sometimes.
    assert float(np.mean(result.eval_returns)) > 0.0


def test_offline_training_reports_real_evaluation_returns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.npz"
    step = make_step_fn(_GRID)
    obs, act, rew, nxt, term = [], [], [], [], []
    rng = np.random.default_rng(0)
    state = 0
    for i in range(600):
        action = int(rng.integers(0, 4))
        next_state, reward, done = step(
            jnp.asarray(state), jnp.asarray(action), jax.random.PRNGKey(i)
        )
        obs.append(state)
        act.append(action)
        rew.append(float(reward))
        nxt.append(int(next_state))
        term.append(bool(done))
        state = 0 if bool(done) else int(next_state)
    np.savez(
        dataset_path,
        observations=np.array(obs, np.int32),
        actions=np.array(act, np.int32),
        rewards=np.array(rew, np.float32),
        next_observations=np.array(nxt, np.int32),
        terminals=np.array(term, np.bool_),
    )

    agent = AgentConfig(algorithm="q_learning", discount=0.9, learning_rate=0.3, initial_q=0.0)
    buffer = BufferConfig(
        name="uniform",
        capacity=2000,
        batch_size=32,
        min_size=1,
        updates_per_step=0,
        load_dataset_path=str(dataset_path),
        offline_only=True,
        offline_updates=400,
    )
    result = run_tabular_training(
        agent, _POLICY, _GRID, _runner(train_episodes=4, eval_episodes=5), buffer
    )

    # Fix #2: offline runs used to report all-zero train returns and no eval.
    assert len(result.eval_returns) == 5
    assert result.eval_discounted_returns.shape == result.eval_returns.shape
    assert float(np.mean(result.eval_returns)) > 0.0
    # Discounting still applies to the eval rollouts.
    assert np.all(result.eval_discounted_returns <= result.eval_returns + 1e-6)


def test_riverswim_honours_non_positive_rewards() -> None:
    config = EnvironmentConfig(
        name="riverswim",
        num_states=4,
        num_actions=2,
        start_state=0,
        p_right=0.3,
        p_left=0.1,
        p_stay=0.6,
        easy_reward=-2.0,
        hard_reward=10.0,
        step_reward=0.0,
    )
    # Swimming left (action 0) at the leftmost state yields the easy reward,
    # which used to collapse to step_reward when non-positive.
    _, reward, _ = riverswim_step(config, jnp.asarray(0), jnp.asarray(0), jax.random.PRNGKey(0))
    assert float(reward) == -2.0


def test_epsilon_schedule_tolerates_zero_decay_steps() -> None:
    value = _epsilon(jnp.asarray(0), start=1.0, end=0.05, decay_steps=0)
    assert np.isfinite(float(value))
    assert float(value) == 1.0
