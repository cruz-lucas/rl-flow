from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def replay_buffer_components() -> list[ComponentSpec]:
    output = [PortSpec(name="replay_buffer", type="replay_buffer")]
    return [
        ComponentSpec(
            id="builtin.replay.tabular_uniform",
            source="builtin",
            kind="replay_buffer",
            display_name="Tabular Uniform Replay",
            description="Uniform replay buffer for scalar tabular transitions.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "capacity": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "min_size": {"type": "integer", "minimum": 1},
                    "updates_per_step": {"type": "integer", "minimum": 0},
                    "save_dataset_path": {"type": "string"},
                    "load_dataset_path": {"type": "string"},
                    "offline_only": {"type": "boolean"},
                    "offline_updates": {"type": "integer", "minimum": 0},
                }
            ),
            defaults={
                "capacity": 1024,
                "batch_size": 16,
                "min_size": 32,
                "updates_per_step": 1,
                "save_dataset_path": "",
                "load_dataset_path": "",
                "offline_only": False,
                "offline_updates": 0,
            },
        ),
        ComponentSpec(
            id="builtin.replay.uniform",
            source="builtin",
            kind="replay_buffer",
            display_name="Uniform Replay",
            description="Uniform replay buffer for vector observations used by builtin DQN agents.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "capacity": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "min_size": {"type": "integer", "minimum": 1},
                    "updates_per_step": {"type": "integer", "minimum": 1},
                    "save_dataset_path": {"type": "string"},
                }
            ),
            defaults={
                "capacity": 10000,
                "batch_size": 32,
                "min_size": 500,
                "updates_per_step": 1,
                "save_dataset_path": "",
            },
        ),
    ]
