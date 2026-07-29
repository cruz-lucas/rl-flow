"""Tests for the Strehl & Littman exploration additions: count-based intrinsic
reward, MBIE-EB, and the replay-based optimistic Q-learning agents (Algorithm 1)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rlflow_builtin.tabular.buffers import buffer_config, no_buffer_config
from rlflow_builtin.tabular.environments import environment_config
from rlflow_builtin.tabular.training import (
    _apply_count_bonus,
    _initial_rmax_model,
    _known_mask,
    _mbie_observe_transition,
    _optimistic_bonus,
    _optimistic_values,
    _update_optimistic_transition,
    run_tabular_training,
)
from rlflow_builtin.tabular.types import (
    AgentConfig,
    EnvironmentConfig,
    RunnerConfig,
    TransitionBatch,
)

S = lambda x: jnp.asarray(x, dtype=jnp.int32)  # noqa: E731
F = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731


def _deterministic_chain(num_states: int = 4) -> EnvironmentConfig:
    """RiverSwim with deterministic rightward current: right always advances, the far
    right pays ``hard_reward`` and the left pays a small ``easy_reward``. Optimal is to
    swim right, which a non-exploring greedy learner never discovers."""
    return environment_config(
        "builtin.env.riverswim",
        {
            "num_states": num_states,
            "start_state": 0,
            "random_start": False,
            "p_left": 0.0,
            "p_stay": 0.0,
            "p_right": 1.0,
            "easy_reward": 1.0,
            "hard_reward": 100.0,
            "common_reward": 0.0,
        },
    )


def _runner(steps: int = 400) -> RunnerConfig:
    return RunnerConfig(
        seed=0,
        train_episodes=1,
        train_steps=steps,
        max_episode_steps=steps,
        eval_episodes=0,
        checkpoint_freq=None,
        checkpoint_dir="checkpoints",
        save_final_checkpoint=False,
    )


def _replay_buffer(**overrides):
    config = {
        "capacity": 500,
        "batch_size": 32,
        "min_size": 1,
        "updates_per_step": 1,
        "save_dataset_path": "",
        "load_dataset_path": "",
        "offline_only": False,
        "offline_updates": 0,
        "replay_until_convergence": True,
        "convergence_tol": 1e-2,
        "max_replay_iters": 60,
    }
    config.update(overrides)
    return buffer_config("builtin.replay.tabular_uniform", config)


def _cumulative(result) -> float:
    return float(np.sum(result.train_returns))


# --------------------------------------------------------------------------- #
# Count-based intrinsic reward
# --------------------------------------------------------------------------- #
def test_apply_count_bonus_zero_beta_is_noop() -> None:
    agent = AgentConfig(algorithm="q_learning", count_bonus_beta=0.0)
    counts = jnp.ones((2, 2), dtype=jnp.float32)
    reward = F(3.0)
    out = _apply_count_bonus(agent, reward, counts, S(0), S(1))
    assert float(out) == 3.0


def test_apply_count_bonus_positive_beta_adds_inverse_sqrt_count() -> None:
    agent = AgentConfig(algorithm="q_learning", count_bonus_beta=2.0)
    counts = jnp.zeros((2, 2), dtype=jnp.float32).at[0, 1].set(4.0)
    # bonus = 2 / sqrt(max(4, 1)) = 1.0
    out = _apply_count_bonus(agent, F(3.0), counts, S(0), S(1))
    np.testing.assert_allclose(float(out), 4.0, rtol=1e-6)
    # unvisited pair clamps N to 1: bonus = 2 / sqrt(1) = 2.0
    out0 = _apply_count_bonus(agent, F(0.0), counts, S(1), S(0))
    np.testing.assert_allclose(float(out0), 2.0, rtol=1e-6)


def test_count_bonus_flows_into_online_td_target() -> None:
    """The intrinsic bonus is added to the online TD target, so a visited (s,a) ends
    up with a higher value than the same run without the bonus. (Whether that produces
    *exploration* depends on the other factors -- epsilon-greedy / optimistic init.)"""
    from rlflow_builtin.tabular.types import PolicyConfig

    env = _deterministic_chain()
    policy = PolicyConfig(name="epsilon_greedy", epsilon=0.0, eval_epsilon=0.0)
    plain = AgentConfig(algorithm="q_learning", learning_rate=0.5, discount=0.9, initial_q=0.0)
    bonus = AgentConfig(
        algorithm="q_learning",
        learning_rate=0.5,
        discount=0.9,
        initial_q=0.0,
        count_bonus_beta=50.0,
    )
    plain_result = run_tabular_training(plain, policy, env, _runner(), no_buffer_config())
    bonus_result = run_tabular_training(bonus, policy, env, _runner(), no_buffer_config())
    plain_q = np.asarray(plain_result.q_table)
    bonus_q = np.asarray(bonus_result.q_table)
    visited = np.asarray(plain_result.action_counts) > 0
    # Extrinsic behaviour (and thus visited pairs) is identical; the bonus inflates
    # the value of every visited state-action.
    assert np.all(bonus_q[visited] > plain_q[visited] + 1e-3)


# --------------------------------------------------------------------------- #
# MBIE-EB
# --------------------------------------------------------------------------- #
def test_mbie_observe_marks_known_after_one_visit_with_bonus() -> None:
    agent = AgentConfig(
        algorithm="mbie_eb",
        discount=0.9,
        mbie_beta=2.0,
        rmax_v_max=5.0,
        planning_iterations=1,
    )
    env = EnvironmentConfig(name="gridworld", num_states=2, num_actions=2, start_state=0)
    model = _initial_rmax_model(agent, env)
    assert np.allclose(np.asarray(model.q_table), 5.0)  # optimistic init

    model, _ = _mbie_observe_transition(agent, model, S(0), S(0), F(1.0), S(1), jnp.asarray(False))

    counts = np.asarray(model.model_counts)
    q = np.asarray(model.q_table)
    assert counts[0, 0] == 1.0
    # (0,0) is now known: target = mean_r(1) + beta/sqrt(1)(2) + gamma(0.9)*V(s'=1)(=5) = 7.5
    np.testing.assert_allclose(q[0, 0], 7.5, rtol=1e-6)
    # unvisited pairs stay optimistic
    assert q[0, 1] == 5.0
    assert q[1, 0] == 5.0


def test_mbie_eb_solves_deterministic_chain() -> None:
    env = _deterministic_chain()
    agent = AgentConfig(
        algorithm="mbie_eb",
        discount=0.9,
        mbie_beta=10.0,
        rmax_v_max=1000.0,
        planning_iterations=100,
    )
    result = run_tabular_training(agent, None, env, _runner(), no_buffer_config())
    # Reaches the far-right hard reward (100/step) rather than the easy corner (1/step).
    assert _cumulative(result) > 5000.0
    # Optimal policy: pick "right" (action 1) at the start state.
    assert int(np.argmax(np.asarray(result.q_table)[0])) == 1


def test_mbie_eb_rejects_replay_buffer() -> None:
    env = _deterministic_chain()
    agent = AgentConfig(algorithm="mbie_eb", discount=0.9, mbie_beta=1.0, rmax_v_max=10.0)
    with pytest.raises(ValueError, match="does not support replay buffers"):
        run_tabular_training(agent, None, env, _runner(20), _replay_buffer())


# --------------------------------------------------------------------------- #
# Replay-based optimistic Q-learning (Algorithm 1)
# --------------------------------------------------------------------------- #
def test_known_mask_variant_thresholds() -> None:
    counts = F([0.0, 1.0, 2.0, 3.0])
    rmax = AgentConfig(algorithm="replay_rmax", known_count_threshold=2)
    mbie = AgentConfig(algorithm="replay_mbie_eb")
    np.testing.assert_array_equal(np.asarray(_known_mask(rmax, counts)), [False, False, True, True])
    np.testing.assert_array_equal(np.asarray(_known_mask(mbie, counts)), [False, True, True, True])


def test_optimistic_values_use_vmax_for_unknown() -> None:
    agent = AgentConfig(algorithm="replay_mbie_eb", rmax_v_max=9.0)
    q = F([[1.0, 2.0], [3.0, 4.0]])
    counts = F([[0.0, 5.0], [0.0, 0.0]])
    u = np.asarray(_optimistic_values(agent, q, counts))
    np.testing.assert_array_equal(u, [[9.0, 2.0], [9.0, 9.0]])


def test_update_optimistic_transition_gates_unknown_pairs() -> None:
    agent = AgentConfig(
        algorithm="replay_rmax",
        learning_rate=1.0,
        discount=0.9,
        known_count_threshold=2,
        rmax_v_max=10.0,
    )
    q = jnp.zeros((2, 2), dtype=jnp.float32)
    transition = TransitionBatch(
        observations=S(0),
        actions=S(0),
        rewards=F(1.0),
        next_observations=S(1),
        terminals=jnp.asarray(False),
    )
    # (0,0) unknown (N=1 < m=2): predicate false -> Q must not change.
    counts_unknown = jnp.zeros((2, 2), dtype=jnp.float32).at[0, 0].set(1.0)
    (q_out, _), loss = _update_optimistic_transition(
        agent, counts_unknown, (q, jnp.asarray(0)), transition, 2
    )
    assert float(q_out[0, 0]) == 0.0
    assert float(loss) == 0.0

    # (0,0) known (N=2): target = r(1) + gamma(0.9) * max U(s'=1)=Vmax(10) = 10.
    counts_known = jnp.zeros((2, 2), dtype=jnp.float32).at[0, 0].set(2.0)
    (q_out, _), loss = _update_optimistic_transition(
        agent, counts_known, (q, jnp.asarray(0)), transition, 2
    )
    np.testing.assert_allclose(float(q_out[0, 0]), 10.0, rtol=1e-6)
    assert float(loss) > 0.0


def test_update_optimistic_transition_mbie_adds_bonus() -> None:
    agent = AgentConfig(
        algorithm="replay_mbie_eb",
        learning_rate=1.0,
        discount=0.9,
        mbie_beta=2.0,
        rmax_v_max=10.0,
    )
    q = jnp.zeros((2, 2), dtype=jnp.float32)
    counts = jnp.zeros((2, 2), dtype=jnp.float32).at[0, 0].set(1.0)
    transition = TransitionBatch(
        observations=S(0),
        actions=S(0),
        rewards=F(1.0),
        next_observations=S(1),
        terminals=jnp.asarray(False),
    )
    # target = r(1) + beta/sqrt(1)(2) + gamma(0.9)*Vmax(10) = 12.
    (q_out, _), _ = _update_optimistic_transition(agent, counts, (q, jnp.asarray(0)), transition, 2)
    np.testing.assert_allclose(float(q_out[0, 0]), 12.0, rtol=1e-6)


def test_optimistic_bonus_zero_for_rmax_variant() -> None:
    rmax = AgentConfig(algorithm="replay_rmax", mbie_beta=0.0)
    counts = jnp.ones((2, 2), dtype=jnp.float32)
    assert float(_optimistic_bonus(rmax, counts, S(0), S(0))) == 0.0


@pytest.mark.parametrize("algorithm", ["replay_rmax", "replay_mbie_eb"])
def test_replay_optimistic_solves_deterministic_chain(algorithm: str) -> None:
    env = _deterministic_chain()
    agent = AgentConfig(
        algorithm=algorithm,
        learning_rate=0.5,
        discount=0.9,
        known_count_threshold=1,
        mbie_beta=10.0 if algorithm == "replay_mbie_eb" else 0.0,
        rmax_v_max=1000.0,
    )
    result = run_tabular_training(agent, None, env, _runner(), _replay_buffer())
    assert _cumulative(result) > 5000.0
    # Greedy on the learned optimistic values should prefer swimming right.
    u = np.asarray(
        _optimistic_values(agent, jnp.asarray(result.q_table), jnp.asarray(result.action_counts))
    )
    assert int(np.argmax(u[0])) == 1


@pytest.mark.parametrize("algorithm", ["replay_rmax", "replay_mbie_eb"])
def test_replay_optimistic_requires_replay_buffer(algorithm: str) -> None:
    env = _deterministic_chain()
    agent = AgentConfig(algorithm=algorithm, learning_rate=0.5, discount=0.9, rmax_v_max=10.0)
    with pytest.raises(ValueError, match="require a builtin.replay.tabular_uniform"):
        run_tabular_training(agent, None, env, _runner(20), no_buffer_config())
