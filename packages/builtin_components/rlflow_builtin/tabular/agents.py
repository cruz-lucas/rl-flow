from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from rlflow_builtin.tabular.policies import select_action
from rlflow_builtin.tabular.types import AgentConfig, PolicyConfig


def agent_config(component_id: str, config: dict[str, Any]) -> AgentConfig:
    if component_id == "builtin.agent.q_learning_tabular":
        algorithm = "q_learning"
    elif component_id == "builtin.agent.sarsa_tabular":
        algorithm = "sarsa"
    else:
        raise ValueError(f"Unsupported builtin tabular agent: {component_id}")
    return AgentConfig(
        algorithm=algorithm,
        learning_rate=float(config["learning_rate"]),
        discount=float(config["discount"]),
        initial_q=float(config["initial_q"]),
    )


def apply_td_update(
    agent: AgentConfig,
    policy: PolicyConfig,
    q_table: jax.Array,
    action_counts: jax.Array,
    state: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
    key: jax.Array,
    *,
    num_actions: int,
) -> tuple[jax.Array, jax.Array]:
    if agent.algorithm == "sarsa":
        next_action = select_action(
            policy,
            q_table[next_state],
            action_counts[next_state],
            key,
            training=True,
            num_actions=num_actions,
        )
        bootstrap = q_table[next_state, next_action]
    else:
        bootstrap = jnp.max(q_table[next_state])

    terminal_f = terminal.astype(jnp.float32)
    target = reward + agent.discount * bootstrap * (1.0 - terminal_f)
    td_error = target - q_table[state, action]
    updated_q = q_table.at[state, action].add(agent.learning_rate * td_error)
    return updated_q, jnp.abs(td_error)
