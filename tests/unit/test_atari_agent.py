"""Unit tests for the Atari DQN + R-Max + CFN building blocks.

These run on CPU without envpool: the CNN/CFN/R-Max math is exercised directly on
small synthetic tensors, proving the pieces numerically (shapes, determinism,
CFN pseudo-count monotonicity, and that the reused R-Max masks match their
definition on CNN embeddings).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from rlflow_builtin.atari.networks import (
    cfn_encoder_apply,
    cfn_encoder_init,
    nature_cnn_apply,
    nature_cnn_init,
)
from rlflow_builtin.atari.replay import AtariReplay
from rlflow_builtin.dqn.components import _dqn_rmax_defaults, intrinsic_reward_components
from rlflow_builtin.dqn.config import dqn_agent_config, dqn_intrinsic_config
from rlflow_builtin.dqn.intrinsic import (
    _cfn_outputs,
    _cfn_update,
    _initial_intrinsic_state,
    _intrinsic_bonus,
    _intrinsic_bonus_for_all_actions,
)
from rlflow_builtin.dqn.networks import _optimizer
from rlflow_builtin.dqn.policies import _rmax_batch_masks

_OBS = (4, 42, 42)


def _cfn_setup(embed_dim=8, num_actions=4, output_dim=16, conditioning="none"):
    agent = dqn_agent_config("builtin.agent.dqn_rmax_jax", _dqn_rmax_defaults())
    cfn_defaults = next(
        spec for spec in intrinsic_reward_components() if spec.id == "builtin.intrinsic.cfn"
    ).defaults
    cfn_cfg = {
        **cfn_defaults,
        "cfn_output_dim": output_dim,
        "cfn_action_conditioning": conditioning,
    }
    cfn = dqn_intrinsic_config("builtin.intrinsic.cfn", cfn_cfg, agent)
    state = _initial_intrinsic_state(agent, cfn, embed_dim, num_actions, jax.random.PRNGKey(0))
    optimizer = _optimizer(agent, cfn.learning_rate, cfn.optimizer)
    return agent, cfn, state, optimizer


# --- networks ---------------------------------------------------------------


def test_nature_cnn_output_shape_and_finite():
    params = nature_cnn_init(jax.random.PRNGKey(0), 4, 42, 42, 6, head_hidden=(64,))
    obs = jnp.asarray(np.random.default_rng(0).integers(0, 256, (3, *_OBS), dtype=np.uint8))
    q_values = nature_cnn_apply(params, obs)
    assert q_values.shape == (3, 6)
    assert bool(jnp.all(jnp.isfinite(q_values)))


def test_cfn_encoder_is_deterministic_and_nonnegative():
    params = cfn_encoder_init(jax.random.PRNGKey(1), 4, 42, 42, 16)
    obs = jnp.asarray(np.random.default_rng(1).integers(0, 256, (2, *_OBS), dtype=np.uint8))
    first = cfn_encoder_apply(params, obs)
    second = cfn_encoder_apply(params, obs)
    assert first.shape == (2, 16)
    assert bool(jnp.allclose(first, second))
    assert bool(jnp.all(first >= 0.0))


# --- replay -----------------------------------------------------------------


def test_replay_ring_wraps_and_samples_expected_shapes():
    replay = AtariReplay(capacity=10, observation_shape=_OBS, target_dim=8, seed=0)

    def _batch(n, value):
        obs = np.full((n, *_OBS), value, dtype=np.uint8)
        return (
            obs,
            obs.copy(),
            np.zeros(n, dtype=np.int32),
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=np.bool_),
            replay.sample_coin_targets(n, 8),
        )

    replay.add_batch(*_batch(6, 1))
    assert len(replay) == 6
    replay.add_batch(*_batch(6, 2))
    assert len(replay) == 10  # capped at capacity

    sample = replay.sample(4)
    assert sample["observations"].shape == (4, *_OBS)
    assert sample["next_observations"].shape == (4, *_OBS)
    assert sample["actions"].shape == (4,)
    assert set(np.unique(sample["intrinsic_targets"])).issubset({-1.0, 1.0})


def test_coin_targets_are_pm1_and_seed_reproducible():
    left = AtariReplay(4, (4, 8, 8), 8, seed=7).sample_coin_targets(5, 8)
    right = AtariReplay(4, (4, 8, 8), 8, seed=7).sample_coin_targets(5, 8)
    assert np.array_equal(left, right)
    assert set(np.unique(left)).issubset({-1, 1})


# --- CFN pseudo-count behaviour --------------------------------------------


def test_cfn_bonus_is_lower_for_a_repeatedly_seen_state():
    embed_dim, output_dim = 8, 16
    agent, cfn, state, optimizer = _cfn_setup(embed_dim=embed_dim, output_dim=output_dim)
    seen = jnp.tile(jnp.linspace(-1.0, 1.0, embed_dim)[None, :], (32, 1))
    novel = jnp.tile(jnp.linspace(1.0, -1.0, embed_dim)[None, :], (32, 1))
    actions = jnp.zeros((32,), dtype=jnp.int32)
    rng = np.random.default_rng(0)

    for i in range(300):
        targets = jnp.asarray(rng.integers(0, 2, (32, output_dim)).astype(np.float32) * 2.0 - 1.0)
        batch = {
            "observations": seen,
            "actions": actions,
            "intrinsic_targets": targets,
            "rewards": jnp.zeros((32,), dtype=jnp.float32),
        }
        _bonus, state, _loss = _cfn_update(
            state, batch, cfn, optimizer, 4, jnp.asarray(i + 1, dtype=jnp.int32)
        )

    raw_seen = _cfn_outputs(state.prior_params, state.predictor_params, seen, actions, cfn, 4)[0]
    raw_novel = _cfn_outputs(state.prior_params, state.predictor_params, novel, actions, cfn, 4)[0]
    # A frequently-seen state's coin-flips cancel toward zero (low pseudo-count
    # bonus); a novel state retains the random-prior magnitude (high bonus).
    assert float(jnp.mean(raw_seen)) < float(jnp.mean(raw_novel))


def test_rmax_masks_match_their_definition_on_embeddings():
    embed_dim, num_actions = 8, 4
    agent, cfn, state, _optimizer = _cfn_setup(embed_dim=embed_dim, num_actions=num_actions)
    obs = jnp.asarray(np.random.default_rng(2).normal(size=(16, embed_dim)).astype(np.float32))
    next_obs = jnp.asarray(np.random.default_rng(3).normal(size=(16, embed_dim)).astype(np.float32))
    actions = jnp.asarray(
        np.random.default_rng(4).integers(0, num_actions, size=16), dtype=jnp.int32
    )

    known, next_unknown = _rmax_batch_masks(
        state,
        cfn,
        agent,
        {"observations": obs, "next_observations": next_obs, "actions": actions},
        num_actions,
    )
    current_bonus = _intrinsic_bonus(state, obs, actions, cfn, num_actions)
    next_bonus = _intrinsic_bonus_for_all_actions(state, next_obs, cfn, num_actions)
    assert bool(jnp.array_equal(known, current_bonus <= agent.rmax_bonus_threshold))
    assert bool(
        jnp.array_equal(next_unknown, jnp.any(next_bonus > agent.rmax_bonus_threshold, axis=1))
    )
