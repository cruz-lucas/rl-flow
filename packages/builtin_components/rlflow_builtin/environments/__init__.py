"""Built-in environment helpers."""
from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def environment_components() -> list[ComponentSpec]:
    output = [PortSpec(name="environment", type="environment")]
    return [
        ComponentSpec(
            id="navix.env.grid",
            source="navix",
            kind="environment",
            display_name="Navix Grid Environment",
            description="Navix EmptyRoom, DoorKey, and FourRooms tasks with symbolic, RGB, tabular, one-hot, or one-hot feature observations.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "env_name": {
                        "type": "string",
                        "enum": ["empty_room", "doorkey", "four_rooms"],
                        "description": "Navix task: empty_room, doorkey, or four_rooms.",
                    },
                    "size": {
                        "type": "integer",
                        "enum": [5, 6, 8, 16, 19],
                        "description": "Grid side length. FourRooms requires 19.",
                    },
                    "layout": {"type": "string", "enum": ["fixed", "random", "layout1", "layout2", "layout3"]},
                    "observation_mode": {
                        "type": "string",
                        "enum": ["tabular", "one_hot", "state_features", "symbolic", "rgb"],
                    },
                    "action_set": {"type": "string", "enum": ["default", "cardinal"]},
                    "max_steps": {"type": ["integer", "null"], "minimum": 1},
                    "symbolic_distractor": {
                        "type": "string",
                        "enum": [
                            "none",
                            "corner_wall_color",
                            "shared_wall_color",
                            "independent_wall_color",
                        ],
                    },
                }
            ),
            defaults={
                "env_name": "empty_room",
                "size": 5,
                "layout": "fixed",
                "observation_mode": "tabular",
                "action_set": "default",
                "max_steps": None,
                "symbolic_distractor": "none",
            },
            compile_target={
                "gin": {
                    "imports": ["rlflow_builtin.environments.navix"],
                    "static_bindings": {
                        "Runner.create_environment_fn": "@navix.create_navix_environment",
                        "TrainRunner.create_environment_fn": "@navix.create_navix_environment",
                    },
                    "bindings": {
                        "create_navix_environment.env_name": "env_name",
                        "create_navix_environment.size": "size",
                        "create_navix_environment.layout": "layout",
                        "create_navix_environment.observation_mode": "observation_mode",
                        "create_navix_environment.action_set": "action_set",
                        "create_navix_environment.max_steps": "max_steps",
                        "create_navix_environment.symbolic_distractor": "symbolic_distractor",
                    },
                }
            },
        ),
        ComponentSpec(
            id="builtin.env.gridworld",
            source="builtin",
            kind="environment",
            display_name="Gridworld Environment",
            description="Tabular rectangular gridworld with four actions, terminal goal, and optional pits.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "width": {"type": "integer", "minimum": 2},
                    "height": {"type": "integer", "minimum": 2},
                    "start_state": {"type": "integer", "minimum": 0},
                    "goal_state": {"type": "integer", "minimum": 0},
                    "pit_states": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                    "goal_reward": {"type": "number"},
                    "pit_reward": {"type": "number"},
                    "step_reward": {"type": "number"},
                    "slip_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            ),
            defaults={
                "width": 4,
                "height": 4,
                "start_state": 0,
                "goal_state": 15,
                "pit_states": [],
                "goal_reward": 1.0,
                "pit_reward": -1.0,
                "step_reward": -0.01,
                "slip_probability": 0.0,
            },
        ),
        ComponentSpec(
            id="builtin.env.riverswim",
            source="builtin",
            kind="environment",
            display_name="RiverSwim Environment",
            description="Continuing RiverSwim benchmark from Strehl and Littman with risky rightward current.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "num_states": {"type": "integer", "minimum": 3},
                    "start_state": {"type": "integer", "minimum": 0},
                    "random_start": {"type": "boolean"},
                    "p_left": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "p_stay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "p_right": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "easy_reward": {"type": "number"},
                    "hard_reward": {"type": "number"},
                    "common_reward": {"type": "number"},
                }
            ),
            defaults={
                "num_states": 6,
                "start_state": 1,
                "random_start": True,
                "p_left": 0.1,
                "p_stay": 0.6,
                "p_right": 0.3,
                "easy_reward": 5.0,
                "hard_reward": 10000.0,
                "common_reward": 0.0,
            },
        ),
        ComponentSpec(
            id="builtin.env.sixarms",
            source="builtin",
            kind="environment",
            display_name="SixArms Environment",
            description="Continuing seven-state hub-and-arm benchmark with rare high-reward arms.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "success_probabilities": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 6,
                        "maxItems": 6,
                    },
                    "arm_rewards": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 7,
                        "maxItems": 7,
                    },
                }
            ),
            defaults={
                "success_probabilities": [1.0, 0.15, 0.1, 0.05, 0.03, 0.01],
                "arm_rewards": [0.0, 50.0, 133.0, 300.0, 800.0, 1660.0, 6000.0],
            },
        ),
    ]
