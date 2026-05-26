from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from rlflow_builtin.tabular.types import PolicyConfig


def policy_config(component_id: str, config: dict[str, Any]) -> PolicyConfig:
    if component_id == "builtin.policy.epsilon_greedy":
        return PolicyConfig(
            name="epsilon_greedy",
            epsilon=float(config["epsilon"]),
            eval_epsilon=float(config["eval_epsilon"]),
        )
    if component_id == "builtin.policy.ucb":
        return PolicyConfig(
            name="ucb",
            coefficient=float(config["coefficient"]),
            initial_count=float(config["initial_count"]),
        )
    if component_id == "builtin.policy.softmax":
        return PolicyConfig(
            name="softmax",
            temperature=float(config["temperature"]),
            eval_temperature=float(config["eval_temperature"]),
        )
    raise ValueError(f"Unsupported builtin tabular policy: {component_id}")


def select_action(
    policy: PolicyConfig,
    q_values: jax.Array,
    counts: jax.Array,
    key: jax.Array,
    *,
    training: bool,
    num_actions: int,
) -> jax.Array:
    if policy.name == "epsilon_greedy":
        epsilon = policy.epsilon if training else policy.eval_epsilon
        return epsilon_greedy(q_values, key, epsilon, num_actions)
    if policy.name == "ucb" and training:
        return ucb(q_values, counts, policy.coefficient, policy.initial_count)
    if policy.name == "softmax":
        temperature = policy.temperature if training else policy.eval_temperature
        return softmax(q_values, key, temperature)
    return greedy(q_values, key, num_actions)


def epsilon_greedy(q_values: jax.Array, key: jax.Array, epsilon: float, num_actions: int) -> jax.Array:
    greedy_key, random_key, choice_key = jax.random.split(key, 3)
    random_action = jax.random.randint(random_key, (), 0, num_actions, dtype=jnp.int32)
    greedy_action = greedy(q_values, greedy_key, num_actions)
    explore = jax.random.uniform(choice_key) < epsilon
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def ucb(q_values: jax.Array, counts: jax.Array, coefficient: float, initial_count: float) -> jax.Array:
    total_count = jnp.sum(counts)
    bonus = coefficient * jnp.sqrt(jnp.log(total_count + 1.0) / (counts + initial_count))
    return jnp.argmax(q_values + bonus).astype(jnp.int32)


def softmax(q_values: jax.Array, key: jax.Array, temperature: float) -> jax.Array:
    return jax.random.categorical(key, q_values / temperature).astype(jnp.int32)


def greedy(q_values: jax.Array, key: jax.Array, num_actions: int) -> jax.Array:
    del num_actions
    max_value = jnp.max(q_values)
    mask = q_values == max_value
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits).astype(jnp.int32)
