from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def dqn_agent_components() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            id="builtin.agent.dqn_jax",
            source="builtin",
            kind="agent",
            display_name="DQN JAX",
            description="Standalone JAX DQN agent with an MLP Q-network and target network.",
            output_ports=[PortSpec(name="agent", type="agent")],
            config_schema=component_schema(_dqn_properties()),
            defaults=_dqn_defaults(),
        ),
        ComponentSpec(
            id="builtin.agent.dqn_rmax_jax",
            source="builtin",
            kind="agent",
            display_name="DQN + R-Max JAX",
            description=(
                "DQN agent that treats high intrinsic-bonus state-action pairs as "
                "unknown and assigns them an optimistic R-Max value."
            ),
            output_ports=[PortSpec(name="agent", type="agent")],
            config_schema=component_schema(_dqn_rmax_properties()),
            defaults=_dqn_rmax_defaults(),
        ),
    ]


def intrinsic_reward_components() -> list[ComponentSpec]:
    output = [PortSpec(name="intrinsic_reward", type="intrinsic_reward")]
    return [
        ComponentSpec(
            id="builtin.intrinsic.rnd",
            source="builtin",
            kind="intrinsic_reward",
            display_name="RND Intrinsic Reward",
            description="Random Network Distillation bonus module for DQN.",
            output_ports=output,
            config_schema=component_schema(
                {
                    **_common_intrinsic_properties("rnd"),
                    "rnd_action_conditioning": _action_conditioning_schema(),
                    "rnd_include_action": {"type": ["boolean", "null"]},
                    "rnd_output_dim": {"type": "integer", "minimum": 1},
                    "rnd_update_period": {"type": "integer", "minimum": 1},
                }
            ),
            defaults={
                **_common_intrinsic_defaults("rnd"),
                "rnd_action_conditioning": "none",
                "rnd_include_action": None,
                "rnd_output_dim": 64,
                "rnd_update_period": 1,
            },
        ),
        ComponentSpec(
            id="builtin.intrinsic.cfn",
            source="builtin",
            kind="intrinsic_reward",
            display_name="CFN Intrinsic Reward",
            description="Coin Flip Network bonus module for DQN with replay-stored targets.",
            output_ports=output,
            config_schema=component_schema(
                {
                    **_common_intrinsic_properties("cfn"),
                    "cfn_action_conditioning": _action_conditioning_schema(),
                    "cfn_output_dim": {"type": "integer", "minimum": 1},
                    "cfn_update_period": {"type": "integer", "minimum": 1},
                    "cfn_use_random_prior": {"type": "boolean"},
                    "cfn_prior_scale": {"type": "number", "minimum": 0.0},
                    "cfn_bonus_exponent": {"type": "number", "exclusiveMinimum": 0.0},
                    "cfn_final_tanh": {"type": "boolean"},
                }
            ),
            defaults={
                **_common_intrinsic_defaults("cfn"),
                "cfn_action_conditioning": "output",
                "cfn_output_dim": 64,
                "cfn_update_period": 1,
                "cfn_use_random_prior": True,
                "cfn_prior_scale": 1.0,
                "cfn_bonus_exponent": 0.5,
                "cfn_final_tanh": False,
            },
        ),
        ComponentSpec(
            id="builtin.intrinsic.count",
            source="builtin",
            kind="intrinsic_reward",
            display_name="Count-Based Intrinsic Reward",
            description="Exact count-based exploration bonus module for DQN.",
            output_ports=output,
            config_schema=component_schema(
                {
                    **_common_intrinsic_properties("count"),
                    "count_action_conditioning": _action_conditioning_schema(),
                    "count_key_mode": {
                        "type": "string",
                        "enum": ["dense_exact", "oracle_tabular"],
                    },
                    "count_table_size": {"type": "integer", "minimum": 0},
                    "count_table_overflow": {"type": "string", "enum": ["warn", "error"]},
                    "count_bonus_exponent": {"type": "number", "exclusiveMinimum": 0.0},
                    "count_min_count": {"type": "number", "exclusiveMinimum": 0.0},
                    "count_ignore_empty_room_distractor": {"type": "boolean"},
                }
            ),
            defaults={
                **_common_intrinsic_defaults("count"),
                "intrinsic_stats_decay": 1.0,
                "count_action_conditioning": "input",
                "count_key_mode": "dense_exact",
                "count_table_size": 16384,
                "count_table_overflow": "warn",
                "count_bonus_exponent": 0.5,
                "count_min_count": 1.0,
                "count_ignore_empty_room_distractor": False,
            },
        ),
        ComponentSpec(
            id="builtin.intrinsic.simhash",
            source="builtin",
            kind="intrinsic_reward",
            display_name="SimHash Intrinsic Reward",
            description=(
                "SimHash count-based exploration bonus with static random "
                "projections or learned autoencoder features."
            ),
            output_ports=output,
            config_schema=component_schema(
                {
                    **_common_intrinsic_properties("simhash"),
                    "simhash_mode": {
                        "type": "string",
                        "enum": ["static", "learned", "autoencoder"],
                    },
                    "simhash_action_conditioning": _action_conditioning_schema(),
                    "simhash_bits": {"type": "integer", "minimum": 1},
                    "simhash_latent_dim": {"type": "integer", "minimum": 1},
                    "simhash_table_size": {"type": "integer", "minimum": 1},
                    "simhash_table_overflow": {"type": "string", "enum": ["warn", "error"]},
                    "simhash_bonus_exponent": {"type": "number", "exclusiveMinimum": 0.0},
                    "simhash_min_count": {"type": "number", "exclusiveMinimum": 0.0},
                    "simhash_update_period": {"type": "integer", "minimum": 1},
                    "simhash_ignore_empty_room_distractor": {"type": "boolean"},
                }
            ),
            defaults={
                **_common_intrinsic_defaults("simhash"),
                "intrinsic_stats_decay": 1.0,
                "simhash_mode": "static",
                "simhash_action_conditioning": "input",
                "simhash_bits": 32,
                "simhash_latent_dim": 64,
                "simhash_table_size": 16384,
                "simhash_table_overflow": "warn",
                "simhash_bonus_exponent": 0.5,
                "simhash_min_count": 1.0,
                "simhash_update_period": 1,
                "simhash_ignore_empty_room_distractor": False,
            },
        ),
    ]


def _dqn_properties() -> dict:
    hidden_schema = _hidden_units_schema(allow_null=False)
    optional_hidden_schema = {
        "oneOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "integer", "minimum": 1}},
        ]
    }
    return {
        "learning_rate": {"type": "number", "exclusiveMinimum": 0.0},
        "discount": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "hidden_units": hidden_schema,
        "hidden_dims": _hidden_property(optional_hidden_schema),
        "activation": {"type": "string", "enum": ["relu", "tanh", "gelu", "elu", "linear"]},
        "normalization": _hidden_property({"type": "string"}),
        "update_frequency": {"type": "integer", "minimum": 1},
        "target_update_frequency": {"type": "integer", "minimum": 1},
        "target_update_freq": _hidden_property({"type": "integer", "minimum": 1}),
        "epsilon_start": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "epsilon_end": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "epsilon_decay_steps": {"type": "integer", "minimum": 1},
        "eps_start": _hidden_property({"type": "number", "minimum": 0.0, "maximum": 1.0}),
        "eps_end": _hidden_property({"type": "number", "minimum": 0.0, "maximum": 1.0}),
        "eps_decay_steps": _hidden_property({"type": "integer", "minimum": 1}),
        "eval_epsilon": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "loss_type": {"type": "string", "enum": ["mse", "huber"]},
        "huber_delta": {"type": "number", "exclusiveMinimum": 0.0},
        "double_q": {"type": "boolean"},
        "max_grad_norm": {"type": "number", "minimum": 0.0},
        "optimizer": {"type": "string", "enum": ["adam", "sgd", "rmsprop"]},
        "optimizer_beta1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "optimizer_beta2": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "optimizer_epsilon": {"type": "number", "exclusiveMinimum": 0.0},
        "optimizer_weight_decay": {"type": "number", "minimum": 0.0},
        "optimizer_momentum": {"type": "number", "minimum": 0.0},
        "optimizer_decay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "optimizer_centered": {"type": "boolean"},
        "normalize_observations": {"type": "boolean"},
        "obs_normalization_epsilon": {"type": "number", "exclusiveMinimum": 0.0},
        "obs_normalization_clip": {"type": ["number", "null"], "exclusiveMinimum": 0.0},
        "seed": {"type": "integer", "minimum": 0},
    }


def _dqn_defaults() -> dict:
    return {
        "learning_rate": 0.001,
        "discount": 0.99,
        "hidden_units": [128, 128],
        "activation": "relu",
        "update_frequency": 1,
        "target_update_frequency": 200,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 10000,
        "eval_epsilon": 0.0,
        "loss_type": "mse",
        "huber_delta": 1.0,
        "double_q": False,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_epsilon": 1e-8,
        "optimizer_weight_decay": 0.0,
        "optimizer_momentum": 0.0,
        "optimizer_decay": 0.95,
        "optimizer_centered": False,
        "normalize_observations": False,
        "obs_normalization_epsilon": 1e-8,
        "obs_normalization_clip": 5.0,
        "seed": 0,
    }


def _dqn_rmax_properties() -> dict:
    properties = {
        **_dqn_properties(),
        "rmax_bonus_threshold": {"type": "number", "minimum": 0.0},
        "rmax_decision_v_max": {"type": "number"},
        "rmax_update_v_max": {"type": "number"},
        "rmax_v_max": _hidden_property({"type": "number", "deprecated": True}),
    }
    for key in ("epsilon_start", "epsilon_end", "epsilon_decay_steps", "eval_epsilon"):
        properties[key] = _hidden_property(properties[key])
    return properties


def _dqn_rmax_defaults() -> dict:
    return {
        **_dqn_defaults(),
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "eval_epsilon": 0.0,
        "rmax_bonus_threshold": 0.5,
        "rmax_decision_v_max": 100.0,
        "rmax_update_v_max": 100.0,
        "rmax_v_max": 100.0,
    }


def _common_intrinsic_properties(prefix: str) -> dict:
    return {
        "intrinsic_reward_scale": {"type": "number", "minimum": 0.0},
        "intrinsic_stats_decay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "intrinsic_reward_epsilon": {"type": "number", "exclusiveMinimum": 0.0},
        "intrinsic_reward_clip": {"type": ["number", "null"], "exclusiveMinimum": 0.0},
        "intrinsic_reward_center": {"type": "boolean"},
        f"{prefix}_hidden_units": _hidden_units_schema(allow_null=True),
        f"{prefix}_hidden_dims": _hidden_property(
            {
                "oneOf": [
                    {"type": "null"},
                    {"type": "array", "items": {"type": "integer", "minimum": 1}},
                ]
            }
        ),
        f"{prefix}_activation": {"type": ["string", "null"], "enum": ["relu", "tanh", "gelu", "elu", "linear", None]},
        f"{prefix}_normalization": _hidden_property({"type": ["string", "null"]}),
        f"{prefix}_learning_rate": {"type": ["number", "null"], "exclusiveMinimum": 0.0},
        f"{prefix}_optimizer": {"type": ["string", "null"], "enum": ["adam", "sgd", "rmsprop", None]},
        "debug": {"type": "boolean"},
        "debug_log_dir": {"type": "string"},
        "debug_log_to_mlflow": {"type": "boolean"},
        "debug_compact_observations": {"type": "boolean"},
    }


def _common_intrinsic_defaults(prefix: str) -> dict:
    return {
        "intrinsic_reward_scale": 1.0,
        "intrinsic_stats_decay": 0.99,
        "intrinsic_reward_epsilon": 1e-4,
        "intrinsic_reward_clip": 10.0,
        "intrinsic_reward_center": False,
        f"{prefix}_hidden_units": None,
        f"{prefix}_activation": None,
        f"{prefix}_learning_rate": None,
        f"{prefix}_optimizer": None,
        "debug": False,
        "debug_log_dir": "tmp/debug_logs",
        "debug_log_to_mlflow": True,
        "debug_compact_observations": True,
    }


def _action_conditioning_schema() -> dict:
    return {
        "type": "string",
        "enum": [
            "none",
            "state",
            "observation",
            "input",
            "action_input",
            "include_action",
            "output",
            "action_output",
            "per_action",
            "pair",
            "state_action",
            "state_action_pair",
            "onehot_pair",
            "obs_action_onehot",
        ],
    }


def _hidden_property(schema: dict) -> dict:
    return {**schema, "deprecated": True, "x-inspector-hidden": True}


def _hidden_units_schema(*, allow_null: bool) -> dict:
    options = [
        {"type": "integer", "minimum": 1},
        {"type": "array", "items": {"type": "integer", "minimum": 1}},
        {"type": "string", "pattern": r"^\s*\d+(\s*,\s*\d+)*\s*$"},
    ]
    if allow_null:
        options.insert(0, {"type": "null"})
    return {"oneOf": options}
