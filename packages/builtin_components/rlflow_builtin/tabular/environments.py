from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from rlflow_builtin.tabular.types import EnvironmentConfig


def environment_config(component_id: str, config: dict[str, Any]) -> EnvironmentConfig:
    if component_id == "builtin.env.gridworld":
        return gridworld_environment(config)
    if component_id == "builtin.env.riverswim":
        return riverswim_environment(config)
    if component_id == "builtin.env.sixarms":
        return sixarms_environment(config)
    if component_id == "navix.env.grid":
        return navix_environment(config)
    raise ValueError(f"Unsupported builtin tabular environment: {component_id}")


def gridworld_environment(config: dict[str, Any]) -> EnvironmentConfig:
    width = int(config["width"])
    height = int(config["height"])
    num_states = width * height
    start_state = int(config["start_state"])
    goal_state = int(config["goal_state"])
    pit_states = tuple(int(item) for item in config["pit_states"])
    for name, state in {"start_state": start_state, "goal_state": goal_state}.items():
        if state < 0 or state >= num_states:
            raise ValueError(f"builtin.env.gridworld {name} is outside the grid")
    if any(state < 0 or state >= num_states for state in pit_states):
        raise ValueError("builtin.env.gridworld pit_states contains a state outside the grid")
    return EnvironmentConfig(
        name="gridworld",
        num_states=num_states,
        num_actions=4,
        start_state=start_state,
        width=width,
        height=height,
        goal_state=goal_state,
        pit_states=pit_states,
        goal_reward=float(config["goal_reward"]),
        pit_reward=float(config["pit_reward"]),
        step_reward=float(config["step_reward"]),
        slip_probability=float(config["slip_probability"]),
    )


def riverswim_environment(config: dict[str, Any]) -> EnvironmentConfig:
    num_states = int(config["num_states"])
    if num_states < 3:
        raise ValueError("builtin.env.riverswim num_states must be at least 3")

    p_left = float(config["p_left"])
    p_stay = float(config["p_stay"])
    p_right = float(config["p_right"])
    if not np_isclose(p_left + p_stay + p_right, 1.0):
        raise ValueError("builtin.env.riverswim transition probabilities must sum to 1.0")

    random_start = bool(config["random_start"])
    start_state = int(config["start_state"])
    if start_state < 0 or start_state >= num_states:
        raise ValueError("builtin.env.riverswim start_state is outside the chain")

    random_start_states = tuple(state for state in (1, 2) if state < num_states)
    return EnvironmentConfig(
        name="riverswim",
        num_states=num_states,
        num_actions=2,
        start_state=start_state,
        random_start_states=random_start_states if random_start else (),
        p_left=p_left,
        p_stay=p_stay,
        p_right=p_right,
        easy_reward=float(config["easy_reward"]),
        hard_reward=float(config["hard_reward"]),
        step_reward=float(config["common_reward"]),
    )


def sixarms_environment(config: dict[str, Any]) -> EnvironmentConfig:
    success_probabilities = tuple(float(item) for item in config["success_probabilities"])
    arm_rewards = tuple(float(item) for item in config["arm_rewards"])
    if len(success_probabilities) != 6:
        raise ValueError("builtin.env.sixarms success_probabilities must contain exactly 6 values")
    if len(arm_rewards) != 7:
        raise ValueError("builtin.env.sixarms arm_rewards must contain exactly 7 values")
    if any(probability < 0.0 or probability > 1.0 for probability in success_probabilities):
        raise ValueError("builtin.env.sixarms success probabilities must be between 0 and 1")
    return EnvironmentConfig(
        name="sixarms",
        num_states=7,
        num_actions=6,
        start_state=0,
        success_probabilities=success_probabilities,
        arm_rewards=arm_rewards,
    )


def navix_environment(config: dict[str, Any]) -> EnvironmentConfig:
    if config["observation_mode"] != "tabular":
        raise ValueError("builtin.runner.tabular_jax requires navix.env.grid observation_mode='tabular'")

    from rlflow_builtin.environments.navix import create_navix_environment

    env = create_navix_environment(
        env_name=config["env_name"],
        size=int(config["size"]),
        layout=config["layout"],
        observation_mode=config["observation_mode"],
        action_set=config["action_set"],
        max_steps=config["max_steps"],
    )
    return EnvironmentConfig(
        name="navix",
        num_states=int(env.observation_space.n),
        num_actions=int(env.action_space.n),
        start_state=0,
        navix_env_name=config["env_name"],
        navix_size=int(config["size"]),
        navix_layout=config["layout"],
        navix_observation_mode=config["observation_mode"],
        navix_action_set=config["action_set"],
        navix_max_steps=config["max_steps"],
    )


def initial_state(config: EnvironmentConfig, key: jax.Array) -> jax.Array:
    if config.random_start_states:
        choices = jnp.asarray(config.random_start_states, dtype=jnp.int32)
        return jax.random.choice(key, choices).astype(jnp.int32)
    return jnp.asarray(config.start_state, dtype=jnp.int32)


def make_step_fn(config: EnvironmentConfig) -> Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    if config.name == "navix":
        raise ValueError("Navix environments use the dedicated Navix tabular runner path")
    if config.name == "riverswim":
        return lambda state, action, key: riverswim_step(config, state, action, key)
    if config.name == "sixarms":
        success_probabilities = jnp.asarray(config.success_probabilities, dtype=jnp.float32)
        arm_rewards = jnp.asarray(config.arm_rewards, dtype=jnp.float32)
        return lambda state, action, key: sixarms_step(config, success_probabilities, arm_rewards, state, action, key)
    pit_states = jnp.asarray(config.pit_states, dtype=jnp.int32)
    return lambda state, action, key: gridworld_step(config, pit_states, state, action, key)


def gridworld_step(
    config: EnvironmentConfig,
    pit_states: jax.Array,
    state: jax.Array,
    action: jax.Array,
    key: jax.Array,
):
    action = maybe_slip(action, key, config.num_actions, config.slip_probability)
    row = state // config.width
    col = state % config.width
    next_row = jnp.clip(row + jnp.where(action == 2, 1, jnp.where(action == 0, -1, 0)), 0, config.height - 1)
    next_col = jnp.clip(col + jnp.where(action == 1, 1, jnp.where(action == 3, -1, 0)), 0, config.width - 1)
    next_state = (next_row * config.width + next_col).astype(jnp.int32)
    is_goal = next_state == config.goal_state
    is_pit = jnp.any(next_state == pit_states)
    terminal = jnp.logical_or(is_goal, is_pit)
    reward = jnp.where(
        is_goal,
        config.goal_reward,
        jnp.where(is_pit, config.pit_reward, config.step_reward),
    )
    return next_state, reward.astype(jnp.float32), terminal


def riverswim_step(config: EnvironmentConfig, state: jax.Array, action: jax.Array, key: jax.Array):
    state = state.astype(jnp.int32)
    action = action.astype(jnp.int32)
    last_state = config.num_states - 1
    delta_candidates = jnp.asarray([-1, 0, 1], dtype=jnp.int32)
    interior_right_probabilities = jnp.asarray([config.p_left, config.p_stay, config.p_right], dtype=jnp.float32)
    right_edge_probabilities = jnp.asarray([1.0 - config.p_right, 0.0, config.p_right], dtype=jnp.float32)
    probabilities = jnp.where(state == last_state, right_edge_probabilities, interior_right_probabilities)
    right_delta = jax.random.choice(key, delta_candidates, p=probabilities)
    right_state = jnp.clip(state + right_delta, 0, last_state)
    left_state = jnp.clip(state - 1, 0, last_state)
    next_state = jnp.where(action == 0, left_state, right_state).astype(jnp.int32)

    same_state = state == next_state
    easy_reward = jnp.logical_and(action == 0, state == 0).astype(jnp.float32) * config.easy_reward
    hard_reward = (
        jnp.logical_and(jnp.logical_and(action == 1, same_state), state == last_state).astype(jnp.float32)
        * config.hard_reward
    )
    reward = jnp.where((easy_reward + hard_reward) > 0.0, easy_reward + hard_reward, config.step_reward)
    return next_state, reward.astype(jnp.float32), jnp.asarray(False, dtype=jnp.bool_)


def sixarms_step(
    config: EnvironmentConfig,
    success_probabilities: jax.Array,
    arm_rewards: jax.Array,
    state: jax.Array,
    action: jax.Array,
    key: jax.Array,
):
    del config
    state = state.astype(jnp.int32)
    action = action.astype(jnp.int32)
    success_probability = jnp.take(success_probabilities, action, mode="clip")
    succeeds = jax.random.uniform(key) < success_probability
    next_from_hub = jnp.where(succeeds, action + 1, 0).astype(jnp.int32)

    branch_transitions = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 6],
        ],
        dtype=jnp.int32,
    )
    branch_row = jnp.take(branch_transitions, state, axis=0, mode="clip")
    next_from_branch = jnp.take(branch_row, action, axis=0, mode="clip")
    next_state = jnp.where(state == 0, next_from_hub, next_from_branch).astype(jnp.int32)
    same_state = state == next_state
    reward = jnp.take(arm_rewards, next_state, mode="clip") * same_state.astype(jnp.float32)
    return next_state, reward.astype(jnp.float32), jnp.asarray(False, dtype=jnp.bool_)


def maybe_slip(action: jax.Array, key: jax.Array, num_actions: int, slip_probability: float) -> jax.Array:
    slip_key, action_key = jax.random.split(key)
    slipped_action = jax.random.randint(action_key, (), 0, num_actions, dtype=jnp.int32)
    should_slip = jax.random.uniform(slip_key) < slip_probability
    return jnp.where(should_slip, slipped_action, action).astype(jnp.int32)


def np_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6
