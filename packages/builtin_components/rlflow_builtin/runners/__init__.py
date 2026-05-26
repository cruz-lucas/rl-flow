"""Built-in runner entry points."""
from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def runner_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            id="builtin.runner.tabular_jax",
            source="builtin",
            kind="runner",
            display_name="Builtin JAX Runner",
            description="JAX runner for builtin tabular agents and DQN-family agents.",
            input_ports=[
                PortSpec(name="agent", type="agent"),
                PortSpec(name="environment", type="environment"),
                PortSpec(name="policy", type="policy", required=False),
                PortSpec(name="replay_buffer", type="replay_buffer", required=False),
                PortSpec(name="intrinsic_reward", type="intrinsic_reward", required=False),
            ],
            output_ports=[PortSpec(name="experiment", type="experiment")],
            config_schema=component_schema(
                {
                    "seed": {"type": "integer", "minimum": 0},
                    "train_episodes": {"type": "integer", "minimum": 1},
                    "train_steps": {"type": ["integer", "null"], "minimum": 1},
                    "max_episode_steps": {"type": "integer", "minimum": 1},
                    "eval_episodes": {"type": "integer", "minimum": 0},
                    "checkpoint_freq": {"type": ["integer", "null"], "minimum": 1},
                    "checkpoint_dir": {"type": "string"},
                    "save_final_checkpoint": {"type": "boolean"},
                }
            ),
            defaults={
                "seed": 0,
                "train_episodes": 200,
                "train_steps": None,
                "max_episode_steps": 100,
                "eval_episodes": 20,
                "checkpoint_freq": None,
                "checkpoint_dir": "checkpoints",
                "save_final_checkpoint": True,
            },
            compile_target={"command": {"module": "rlflow_builtin.runners.tabular_jax"}},
        ),
    ]
