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
    elif component_id == "builtin.agent.rmax_tabular":
        return AgentConfig(
            algorithm="rmax",
            discount=float(config["discount"]),
            known_count_threshold=int(config["known_count_threshold"]),
            rmax_v_max=float(config["rmax_v_max"]),
            planning_iterations=int(config["planning_iterations"]),
        )
    elif component_id == "builtin.agent.mbie_eb_tabular":
        return AgentConfig(
            algorithm="mbie_eb",
            discount=float(config["discount"]),
            mbie_beta=float(config["mbie_beta"]),
            rmax_v_max=float(config["rmax_v_max"]),
            planning_iterations=int(config["planning_iterations"]),
            known_count_threshold=1,
        )
    elif component_id == "builtin.agent.replay_rmax_tabular":
        return AgentConfig(
            algorithm="replay_rmax",
            learning_rate=float(config["learning_rate"]),
            discount=float(config["discount"]),
            known_count_threshold=int(config["known_count_threshold"]),
            rmax_v_max=float(config["rmax_v_max"]),
        )
    elif component_id == "builtin.agent.replay_mbie_eb_tabular":
        return AgentConfig(
            algorithm="replay_mbie_eb",
            learning_rate=float(config["learning_rate"]),
            discount=float(config["discount"]),
            mbie_beta=float(config["mbie_beta"]),
            rmax_v_max=float(config["rmax_v_max"]),
            known_count_threshold=1,
        )
    else:
        raise ValueError(
            f"Unsupported builtin tabular agent: {component_id}. Third-party agents "
            "must declare compile_target={'runtime': {'entry_point': 'pkg.module:fn'}} "
            "on their ComponentSpec to be executable by this runner."
        )
    return AgentConfig(
        algorithm=algorithm,
        learning_rate=float(config["learning_rate"]),
        discount=float(config["discount"]),
        initial_q=float(config["initial_q"]),
        count_bonus_beta=float(config.get("count_bonus_beta", 0.0)),
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
    next_action: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if agent.algorithm == "sarsa":
        if next_action is None:
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
