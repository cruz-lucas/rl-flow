"""An out-of-tree example agent used to prove third-party components execute.

This module plays the role of an external plugin package: in a real deployment it
would live in its own installed distribution and expose ``components`` through the
``rlflow.components`` entry-point group::

    [project.entry-points."rlflow.components"]
    example = "example_agent:components"

The agent itself is a uniform-random policy — deliberately trivial so the test
exercises the *dispatch mechanism*, not RL performance. What matters is the
``compile_target`` runtime hook, which makes the component executable by the
builtin runner rather than only renderable in the UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow.schemas.workflow import WorkflowSpec
from rlflow_builtin.tabular.environments import environment_config, initial_state, make_step_fn

EXAMPLE_AGENT_ID = "example.agent.random_tabular"


def components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            id=EXAMPLE_AGENT_ID,
            source="example",
            kind="agent",
            display_name="Example Random Agent",
            description="Uniform-random tabular agent from an out-of-tree package.",
            output_ports=[PortSpec(name="agent", type="agent", required=True)],
            config_schema={
                "type": "object",
                "properties": {
                    "episodes": {"type": "integer", "default": 5},
                },
                "additionalProperties": False,
            },
            defaults={"episodes": 5},
            compile_target={"runtime": {"entry_point": "example_agent:run_training"}},
        )
    ]


def run_training(*, workflow: WorkflowSpec, resolved: dict[str, Any], run_dir: Path) -> None:
    """Train (well: roll a random policy) and write standard run outputs."""
    agent_node = next(node for node in workflow.nodes if node.component == EXAMPLE_AGENT_ID)
    env_node = next(
        node
        for node in workflow.nodes
        if node.component.startswith("builtin.env.") or node.component == "navix.env.grid"
    )
    episodes = int(resolved[agent_node.id].get("episodes", 5))

    environment = environment_config(env_node.component, resolved[env_node.id])
    step = make_step_fn(environment)
    key = jax.random.PRNGKey(0)

    returns: list[float] = []
    lengths: list[int] = []
    for _episode in range(episodes):
        key, reset_key = jax.random.split(key)
        state = initial_state(environment, reset_key)
        episode_return = 0.0
        steps = 0
        for _ in range(20):
            key, action_key, env_key = jax.random.split(key, 3)
            action = jax.random.randint(action_key, (), 0, environment.num_actions)
            state, reward, terminal = step(state, action, env_key)
            episode_return += float(reward)
            steps += 1
            if bool(terminal):
                break
        returns.append(episode_return)
        lengths.append(steps)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_step = 0
    with (logs_dir / "train_history.jsonl").open("w", encoding="utf-8") as handle:
        for episode, (episode_return, length) in enumerate(zip(returns, lengths, strict=True)):
            env_step += length
            handle.write(
                json.dumps(
                    {
                        "episode": episode,
                        "env_step": env_step,
                        "return": episode_return,
                        "length": length,
                        "loss": 0.0,
                    }
                )
                + "\n"
            )

    summaries = run_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    metrics = {
        "agent": EXAMPLE_AGENT_ID,
        "environment": env_node.component,
        "train_episodes": episodes,
        "mean_train_return": float(np.mean(returns)) if returns else None,
    }
    payload = json.dumps(metrics, sort_keys=True)
    (summaries / "metrics.json").write_text(payload, encoding="utf-8")
    (run_dir / "metrics.json").write_text(payload, encoding="utf-8")
