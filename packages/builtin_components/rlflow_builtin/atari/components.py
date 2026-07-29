"""Component specs for the Atari DQN + R-Max + CFN integration.

Import-safe: only pydantic specs + schema helpers, no jax / envpool. The agent's
``compile_target`` runtime hook makes it executable by the builtin runner via the
non-JAX delegate in :mod:`rlflow_builtin.atari.training`.

The agent reuses the ``dqn_rmax_jax`` config surface verbatim (so authoring feels
identical), with defaults retuned for Atari. ``hidden_units`` configures the CNN
dense head. CFN and replay are ordinary ``builtin.intrinsic.cfn`` /
``builtin.replay.uniform`` nodes wired the same way as the fourrooms workflow.
"""

from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema
from rlflow_builtin.dqn.components import _dqn_rmax_defaults, _dqn_rmax_properties

ATARI_ENV_COMPONENT = "builtin.env.atari"
ATARI_AGENT_COMPONENT = "builtin.agent.dqn_rmax_atari"
ATARI_RUNTIME_ENTRY_POINT = "rlflow_builtin.atari.training:run_training"


def atari_components() -> list[ComponentSpec]:
    return [_atari_env_spec(), _atari_agent_spec()]


def _atari_env_spec() -> ComponentSpec:
    return ComponentSpec(
        id=ATARI_ENV_COMPONENT,
        source="builtin",
        kind="environment",
        display_name="Atari (ALE) Environment",
        description=(
            "Vectorised Atari via envpool (Montezuma's Revenge and other ALE games) "
            "with standard preprocessing: frame skip, frame stacking, grayscale, "
            "resize, and sticky actions. Runs through the non-JAX runtime delegate."
        ),
        output_ports=[PortSpec(name="environment", type="environment")],
        config_schema=component_schema(
            {
                "task_id": {"type": "string"},
                "backend": {"type": "string", "enum": ["envpool", "synthetic"]},
                "num_envs": {"type": "integer", "minimum": 1},
                "frame_skip": {"type": "integer", "minimum": 1},
                "frame_stack": {"type": "integer", "minimum": 1},
                "img_height": {"type": "integer", "minimum": 1},
                "img_width": {"type": "integer", "minimum": 1},
                "gray_scale": {"type": "boolean"},
                "repeat_action_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "episodic_life": {"type": "boolean"},
                "noop_max": {"type": "integer", "minimum": 0},
                "max_episode_steps": {"type": ["integer", "null"], "minimum": 1},
                "reward_clip": {"type": "boolean"},
                "full_action_space": {"type": "boolean"},
            }
        ),
        defaults={
            "task_id": "MontezumaRevenge-v5",
            "backend": "envpool",
            "num_envs": 8,
            "frame_skip": 4,
            "frame_stack": 4,
            "img_height": 84,
            "img_width": 84,
            "gray_scale": True,
            "repeat_action_probability": 0.25,
            "episodic_life": False,
            "noop_max": 30,
            "max_episode_steps": None,
            "reward_clip": True,
            "full_action_space": False,
        },
    )


def _atari_agent_spec() -> ComponentSpec:
    return ComponentSpec(
        id=ATARI_AGENT_COMPONENT,
        source="builtin",
        kind="agent",
        display_name="DQN + R-Max (Atari CNN)",
        description=(
            "Nature-CNN DQN agent with R-Max optimism, driven by a CFN knownness "
            "signal, for pixel-based Atari. Executes via a non-JAX runtime delegate "
            "(envpool vector envs + jitted CNN/CFN updates)."
        ),
        input_ports=[
            PortSpec(
                name="knownness_signal",
                type="intrinsic_reward",
                required=False,
                description="Intrinsic module whose bonus is the R-Max knownness signal (CFN).",
            )
        ],
        output_ports=[PortSpec(name="agent", type="agent")],
        config_schema=component_schema(_dqn_rmax_properties()),
        defaults=_atari_agent_defaults(),
        compile_target={"runtime": {"entry_point": ATARI_RUNTIME_ENTRY_POINT}},
    )


def _atari_agent_defaults() -> dict:
    # Atari-tuned overrides on top of the shared dqn_rmax defaults. R-Max optimism
    # drives exploration, so epsilon is small; hidden_units is the CNN dense head.
    return {
        **_dqn_rmax_defaults(),
        "learning_rate": 6.25e-5,
        "discount": 0.99,
        "hidden_units": [512],
        "double_q": True,
        "loss_type": "huber",
        "huber_delta": 1.0,
        "max_grad_norm": 10.0,
        "update_frequency": 4,
        "target_update_frequency": 8000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": 250000,
        "eval_epsilon": 0.001,
        "rmax_bonus_threshold": 0.6,
        "rmax_decision_v_max": 5.0,
        "rmax_update_v_max": 0.5,
        "rmax_v_max": 5.0,
    }
