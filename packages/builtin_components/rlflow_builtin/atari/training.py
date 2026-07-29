"""Non-JAX runtime delegate: DQN + R-Max + CFN on vectorised Atari.

Resolved from the agent node's ``compile_target={"runtime": {"entry_point": ...}}``
by the builtin runner and invoked as ``run_training(workflow, resolved, run_dir)``.
It owns the whole loop and writes the standard run outputs.

Design: envpool steps a vector of envs on the host; the CNN action selection, the
CFN coin-flip update, and the R-Max knownness masking + Q-update are ``jax.jit``-ed
on the accelerator. The CFN + R-Max *math* is reused verbatim from the jitted DQN
package by feeding a fixed-random-CNN **embedding** into the exact same helpers
(``_initial_intrinsic_state`` / ``_cfn_update`` / ``_intrinsic_bonus_for_all_actions``
/ ``_rmax_batch_masks``) that the MLP pipeline uses — so only the conv forward and
the Q-network are new code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlflow.schemas.workflow import WorkflowSpec
from rlflow_builtin.atari.components import ATARI_AGENT_COMPONENT, ATARI_ENV_COMPONENT
from rlflow_builtin.atari.env import atari_env_config, make_atari_env
from rlflow_builtin.atari.networks import (
    cfn_encoder_apply,
    cfn_encoder_init,
    nature_cnn_apply,
    nature_cnn_init,
)
from rlflow_builtin.atari.replay import AtariReplay
from rlflow_builtin.dqn.config import (
    DQN_REPLAY_COMPONENTS,
    DQN_RMAX_AGENT_COMPONENT,
    DqnReplayConfig,
    dqn_agent_config,
    dqn_intrinsic_config,
    dqn_replay_config,
)
from rlflow_builtin.dqn.intrinsic import (
    _cfn_update,
    _initial_intrinsic_state,
    _intrinsic_bonus_for_all_actions,
)
from rlflow_builtin.dqn.networks import _optimizer
from rlflow_builtin.dqn.policies import _epsilon, _max_actions, _rmax_batch_masks

CFN_EMBED_DIM = 512
CFN_INTRINSIC_COMPONENT = "builtin.intrinsic.cfn"


def run_training(*, workflow: WorkflowSpec, resolved: dict[str, Any], run_dir: Path) -> None:
    agent_node = _find_node(workflow, ATARI_AGENT_COMPONENT)
    env_node = _find_node(workflow, ATARI_ENV_COMPONENT)
    runner_node = _find_runner(workflow)
    cfn_node = _knownness_node(workflow, agent_node.id) or _optional_node(
        workflow, {CFN_INTRINSIC_COMPONENT}
    )
    replay_node = _optional_node(workflow, DQN_REPLAY_COMPONENTS)

    agent = dqn_agent_config(DQN_RMAX_AGENT_COMPONENT, resolved[agent_node.id])
    cfn = dqn_intrinsic_config(
        CFN_INTRINSIC_COMPONENT,
        resolved[cfn_node.id] if cfn_node is not None else None,
        agent,
    )
    replay_cfg = (
        dqn_replay_config(replay_node.component, resolved[replay_node.id])
        if replay_node is not None
        else _default_replay_config()
    )
    env_cfg = atari_env_config(resolved[env_node.id])
    total_env_steps = _total_env_steps(resolved[runner_node.id])
    has_cfn = cfn.kind == "cfn"
    target_dim = int(cfn.output_dim)

    seed = int(agent.seed)
    key = jax.random.PRNGKey(seed)

    env = make_atari_env(env_cfg, seed)
    num_actions = env.num_actions
    in_channels, height, width = env.observation_shape

    key, q_key, enc_key, cfn_key = jax.random.split(key, 4)
    q_params = nature_cnn_init(
        q_key, in_channels, height, width, num_actions, tuple(agent.hidden_units)
    )
    target_params = _clone(q_params)
    q_optimizer = _optimizer(agent, agent.learning_rate, agent.optimizer)
    q_opt_state = q_optimizer.init(q_params)

    cfn_encoder_params = cfn_encoder_init(enc_key, in_channels, height, width, CFN_EMBED_DIM)
    cfn_state = _initial_intrinsic_state(agent, cfn, CFN_EMBED_DIM, num_actions, cfn_key)
    cfn_optimizer = _optimizer(agent, cfn.learning_rate, cfn.optimizer)
    cfn_gradient_step = jnp.asarray(0, dtype=jnp.int32)

    replay = AtariReplay(replay_cfg.capacity, env.observation_shape, target_dim, seed=seed)

    def _td_loss(td_error: jax.Array) -> jax.Array:
        if agent.loss_type == "mse":
            return jnp.square(td_error)
        abs_error = jnp.abs(td_error)
        quadratic = jnp.minimum(abs_error, agent.huber_delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + agent.huber_delta * linear

    def _q_loss(params, obs, actions, rewards, next_obs, terminals, known, next_unknown, loss_key):
        q_values = nature_cnn_apply(params, obs, agent.activation)
        selected = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
        if agent.double_q:
            next_online = nature_cnn_apply(params, next_obs, agent.activation)
            next_actions = _max_actions(jax.lax.stop_gradient(next_online), loss_key)[:, None]
            next_target = nature_cnn_apply(target_params, next_obs, agent.activation)
            next_q = jnp.take_along_axis(next_target, next_actions, axis=1).squeeze(-1)
        else:
            next_q = jnp.max(nature_cnn_apply(target_params, next_obs, agent.activation), axis=1)
        next_q = jnp.where(next_unknown, agent.rmax_update_v_max, next_q)
        target = rewards + agent.discount * next_q * (1.0 - terminals)
        td_error = selected - jax.lax.stop_gradient(target)
        losses = _td_loss(td_error)
        weights = known.astype(jnp.float32)
        return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    @jax.jit
    def select_actions(q_params, cfn_state, encoder, obs, act_key, step):
        epsilon = _epsilon(step, agent.epsilon_start, agent.epsilon_end, agent.epsilon_decay_steps)
        q_values = nature_cnn_apply(q_params, obs, agent.activation)
        embedding = cfn_encoder_apply(encoder, obs)
        bonuses = _intrinsic_bonus_for_all_actions(cfn_state, embedding, cfn, num_actions)
        optimistic = jnp.where(
            bonuses > agent.rmax_bonus_threshold, agent.rmax_decision_v_max, q_values
        )
        greedy_key, random_key, choice_key = jax.random.split(act_key, 3)
        greedy = _max_actions(optimistic, greedy_key)
        batch = obs.shape[0]
        random_actions = jax.random.randint(random_key, (batch,), 0, num_actions, dtype=jnp.int32)
        explore = jax.random.uniform(choice_key, (batch,)) < epsilon
        return jnp.where(explore, random_actions, greedy).astype(jnp.int32)

    @jax.jit
    def cfn_update(cfn_state, gradient_step, encoder, obs, actions, targets, rewards):
        embedding = cfn_encoder_apply(encoder, obs)
        batch = {
            "observations": embedding,
            "actions": actions,
            "intrinsic_targets": targets,
            "rewards": rewards,
        }
        _bonus, new_state, loss = _cfn_update(
            cfn_state, batch, cfn, cfn_optimizer, num_actions, gradient_step + 1
        )
        return new_state, gradient_step + 1, loss

    @jax.jit
    def q_update(
        q_params,
        q_opt_state,
        cfn_state,
        encoder,
        obs,
        actions,
        rewards,
        next_obs,
        terminals,
        upd_key,
    ):
        embed_obs = cfn_encoder_apply(encoder, obs)
        embed_next = cfn_encoder_apply(encoder, next_obs)
        known, next_unknown = _rmax_batch_masks(
            cfn_state,
            cfn,
            agent,
            {"observations": embed_obs, "next_observations": embed_next, "actions": actions},
            num_actions,
        )
        loss, grads = jax.value_and_grad(_q_loss)(
            q_params, obs, actions, rewards, next_obs, terminals, known, next_unknown, upd_key
        )
        updates, q_opt_state = q_optimizer.update(grads, q_opt_state, q_params)
        q_params = optax.apply_updates(q_params, updates)
        return q_params, q_opt_state, loss

    obs = env.reset()
    episode_return = np.zeros(env.num_envs, dtype=np.float64)
    episode_length = np.zeros(env.num_envs, dtype=np.int64)
    history: list[dict[str, Any]] = []
    recent_loss = 0.0
    global_step = 0
    last_target_sync = 0
    update_every = max(int(agent.update_frequency), 1)
    num_iterations = max(total_env_steps // env.num_envs, 1)

    for iteration in range(num_iterations):
        key, act_key = jax.random.split(key)
        actions = np.asarray(
            select_actions(
                q_params,
                cfn_state,
                cfn_encoder_params,
                jnp.asarray(obs),
                act_key,
                jnp.asarray(global_step, dtype=jnp.int32),
            )
        )
        next_obs, rewards, terminated, truncated = env.step(actions)
        coin_targets = replay.sample_coin_targets(env.num_envs, target_dim)
        replay.add_batch(obs, next_obs, actions, rewards, terminated, coin_targets)

        episode_return += rewards
        episode_length += 1
        for env_index in np.nonzero(np.logical_or(terminated, truncated))[0]:
            history.append(
                {
                    "episode": len(history),
                    "env_step": global_step + env.num_envs,
                    "return": float(episode_return[env_index]),
                    "length": int(episode_length[env_index]),
                    "loss": float(recent_loss),
                }
            )
            episode_return[env_index] = 0.0
            episode_length[env_index] = 0

        obs = next_obs
        global_step += env.num_envs

        if len(replay) >= replay_cfg.min_size and iteration % update_every == 0:
            if has_cfn:
                for _ in range(int(replay_cfg.intrinsic_updates_per_step or 0)):
                    batch = replay.sample(replay_cfg.batch_size)
                    cfn_state, cfn_gradient_step, _cfn_loss = cfn_update(
                        cfn_state,
                        cfn_gradient_step,
                        cfn_encoder_params,
                        jnp.asarray(batch["observations"]),
                        jnp.asarray(batch["actions"]),
                        jnp.asarray(batch["intrinsic_targets"]),
                        jnp.asarray(batch["rewards"]),
                    )
            for _ in range(int(replay_cfg.q_network_updates_per_step or 0)):
                key, upd_key = jax.random.split(key)
                batch = replay.sample(replay_cfg.batch_size)
                rewards_batch = (
                    np.sign(batch["rewards"]) if env_cfg.reward_clip else batch["rewards"]
                )
                q_params, q_opt_state, loss_value = q_update(
                    q_params,
                    q_opt_state,
                    cfn_state,
                    cfn_encoder_params,
                    jnp.asarray(batch["observations"]),
                    jnp.asarray(batch["actions"]),
                    jnp.asarray(rewards_batch),
                    jnp.asarray(batch["next_observations"]),
                    jnp.asarray(batch["terminals"]),
                    upd_key,
                )
                recent_loss = float(loss_value)

        if global_step - last_target_sync >= agent.target_update_frequency:
            target_params = _clone(q_params)
            last_target_sync = global_step

    _write_outputs(
        run_dir,
        history,
        agent_component=agent_node.component,
        env_component=env_node.component,
        runner_component=runner_node.component,
        replay_component=None if replay_node is None else replay_node.component,
        intrinsic_component=None if cfn_node is None else cfn_node.component,
        total_env_steps=total_env_steps,
    )


def _clone(params):
    return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf, copy=True), params)


def _find_node(workflow: WorkflowSpec, component: str):
    for node in workflow.nodes:
        if node.component == component:
            return node
    raise ValueError(f"Atari workflow is missing a {component!r} node")


def _optional_node(workflow: WorkflowSpec, components: set[str]):
    for node in workflow.nodes:
        if node.component in components:
            return node
    return None


def _find_runner(workflow: WorkflowSpec):
    for node in workflow.nodes:
        if node.component == "builtin.runner.tabular_jax":
            return node
    raise ValueError("Atari workflow requires a builtin.runner.tabular_jax node")


def _knownness_node(workflow: WorkflowSpec, agent_id: str):
    for edge in workflow.edges:
        if edge.to_node == agent_id and edge.to_port == "knownness_signal":
            for node in workflow.nodes:
                if node.id == edge.from_node:
                    return node
    return None


def _default_replay_config() -> DqnReplayConfig:
    return DqnReplayConfig(
        name="builtin.replay.uniform",
        capacity=100_000,
        batch_size=32,
        min_size=1_600,
        updates_per_step=1,
        intrinsic_updates_per_step=1,
        q_network_updates_per_step=1,
    )


def _total_env_steps(runner_settings: dict[str, Any]) -> int:
    steps = runner_settings.get("train_steps")
    if steps:
        return int(steps)
    episodes = runner_settings.get("train_episodes")
    if episodes:
        return int(episodes) * 1000
    return 100_000


def _write_outputs(
    run_dir: Path,
    history: list[dict[str, Any]],
    *,
    agent_component: str,
    env_component: str,
    runner_component: str,
    replay_component: str | None,
    intrinsic_component: str | None,
    total_env_steps: int,
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "train_history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    returns = [row["return"] for row in history]
    summary = {
        "agent": agent_component,
        "environment": env_component,
        "runner": runner_component,
        "replay_buffer": replay_component,
        "intrinsic_reward": intrinsic_component,
        "knownness_signal": intrinsic_component,
        "network": "rlflow_builtin.atari.nature_cnn",
        "agent_algorithm": "dqn_rmax",
        "train_episodes": len(history),
        "train_steps": int(total_env_steps),
        "mean_train_return": float(np.mean(returns)) if returns else None,
        "mean_train_return_last_10": float(np.mean(returns[-10:])) if returns else None,
    }
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True)
    (summaries_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
