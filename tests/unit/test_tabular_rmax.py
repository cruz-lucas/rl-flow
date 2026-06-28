import numpy as np
import jax.numpy as jnp

from rlflow_builtin.tabular.training import _initial_rmax_model, _rmax_observe_transition
from rlflow_builtin.tabular.types import AgentConfig, EnvironmentConfig


def test_tabular_rmax_counts_knownness_and_terminal_bootstrap() -> None:
    agent = AgentConfig(
        algorithm="rmax",
        discount=0.9,
        known_count_threshold=1,
        rmax_v_max=5.0,
        planning_iterations=5,
    )
    environment = EnvironmentConfig(
        name="gridworld",
        num_states=2,
        num_actions=2,
        start_state=0,
    )
    model = _initial_rmax_model(agent, environment)

    assert np.allclose(np.asarray(model.q_table), 5.0)

    model, _ = _rmax_observe_transition(
        agent,
        model,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(2.0, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True),
    )

    counts = np.asarray(model.action_counts)
    q_table = np.asarray(model.q_table)
    transition_counts = np.asarray(model.transition_counts)

    assert counts[0, 0] == 1.0
    assert counts.sum() == 1.0
    assert q_table[0, 0] == 2.0
    assert q_table[0, 1] == 5.0
    assert q_table[1, 0] == 5.0
    assert transition_counts[0, 0].sum() == 0.0


def test_tabular_rmax_nonterminal_transition_bootstraps_from_optimistic_value() -> None:
    agent = AgentConfig(
        algorithm="rmax",
        discount=0.9,
        known_count_threshold=1,
        rmax_v_max=5.0,
        planning_iterations=1,
    )
    environment = EnvironmentConfig(
        name="gridworld",
        num_states=2,
        num_actions=2,
        start_state=0,
    )
    model = _initial_rmax_model(agent, environment)

    model, _ = _rmax_observe_transition(
        agent,
        model,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )

    assert np.asarray(model.action_counts)[0, 0] == 1.0
    assert np.asarray(model.model_counts)[0, 0] == 1.0
    assert np.asarray(model.transition_counts)[0, 0, 1] == 1.0
    np.testing.assert_allclose(np.asarray(model.q_table)[0, 0], 5.5, rtol=1e-6)


def test_tabular_rmax_freezes_model_after_pair_becomes_known() -> None:
    agent = AgentConfig(
        algorithm="rmax",
        discount=0.9,
        known_count_threshold=2,
        rmax_v_max=5.0,
        planning_iterations=1,
    )
    environment = EnvironmentConfig(
        name="gridworld",
        num_states=2,
        num_actions=2,
        start_state=0,
    )
    model = _initial_rmax_model(agent, environment)

    model, _ = _rmax_observe_transition(
        agent,
        model,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )

    assert np.asarray(model.action_counts)[0, 0] == 1.0
    assert np.asarray(model.model_counts)[0, 0] == 1.0
    assert np.asarray(model.q_table)[0, 0] == 5.0

    model, _ = _rmax_observe_transition(
        agent,
        model,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(3.0, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True),
    )

    q_after_known = np.asarray(model.q_table)[0, 0]
    assert np.asarray(model.action_counts)[0, 0] == 2.0
    assert np.asarray(model.model_counts)[0, 0] == 2.0
    assert np.asarray(model.reward_sums)[0, 0] == 4.0
    np.testing.assert_allclose(q_after_known, 4.25, rtol=1e-6)

    model, _ = _rmax_observe_transition(
        agent,
        model,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(100.0, dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True),
    )

    assert np.asarray(model.action_counts)[0, 0] == 3.0
    assert np.asarray(model.model_counts)[0, 0] == 2.0
    assert np.asarray(model.reward_sums)[0, 0] == 4.0
    assert np.asarray(model.transition_counts)[0, 0, 1] == 1.0
    np.testing.assert_allclose(np.asarray(model.q_table)[0, 0], q_after_known, rtol=1e-6)
