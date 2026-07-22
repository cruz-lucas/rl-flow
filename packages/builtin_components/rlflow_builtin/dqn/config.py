"""DQN component configuration: dataclasses, type aliases, and workflow parsers.

Extracted from ``dqn.training`` so the configuration surface is separated from the
training loop. Everything here is dependency-light (no jax/optax) and is
re-exported by ``dqn.training`` for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DQN_AGENT_COMPONENT = "builtin.agent.dqn_jax"
DQN_RMAX_AGENT_COMPONENT = "builtin.agent.dqn_rmax_jax"
DQN_REPLAY_COMPONENTS = {"builtin.replay.uniform", "builtin.replay.dqn_uniform"}

AgentAlgorithm = Literal["dqn", "dqn_rmax"]
IntrinsicKind = Literal["none", "rnd", "cfn", "count", "simhash"]
ActionConditioning = Literal["none", "input", "output", "pair"]
CountTableOverflow = Literal["warn", "error"]
CountKeyMode = Literal["dense_exact", "oracle_tabular"]
SimHashMode = Literal["static", "learned"]


@dataclass(frozen=True)
class DqnAgentConfig:
    algorithm: AgentAlgorithm
    learning_rate: float
    discount: float
    hidden_units: tuple[int, ...]
    activation: str
    update_frequency: int
    target_update_frequency: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    eval_epsilon: float
    loss_type: str
    huber_delta: float
    double_q: bool
    max_grad_norm: float
    optimizer: str
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_epsilon: float
    optimizer_weight_decay: float
    optimizer_momentum: float
    optimizer_decay: float
    optimizer_centered: bool
    normalize_observations: bool
    obs_normalization_epsilon: float
    obs_normalization_clip: float | None
    rmax_bonus_threshold: float
    rmax_decision_v_max: float
    rmax_update_v_max: float
    seed: int


@dataclass(frozen=True)
class DqnReplayConfig:
    name: str
    capacity: int
    batch_size: int
    min_size: int
    updates_per_step: int
    save_dataset_path: str = ""
    intrinsic_updates_per_step: int | None = None
    q_network_updates_per_step: int | None = None


@dataclass(frozen=True)
class DqnIntrinsicConfig:
    kind: IntrinsicKind = "none"
    intrinsic_reward_scale: float = 0.0
    intrinsic_stats_decay: float = 0.99
    intrinsic_reward_epsilon: float = 1e-4
    intrinsic_reward_clip: float | None = 10.0
    intrinsic_reward_center: bool = False
    hidden_units: tuple[int, ...] = ()
    activation: str = "relu"
    output_dim: int = 1
    optimizer: str = "adam"
    learning_rate: float = 0.001
    action_conditioning: ActionConditioning = "none"
    update_period: int = 1
    cfn_use_random_prior: bool = True
    cfn_prior_scale: float = 1.0
    cfn_bonus_exponent: float = 0.5
    cfn_final_tanh: bool = False
    count_table_size: int = 16384
    count_table_overflow: CountTableOverflow = "warn"
    count_key_mode: CountKeyMode = "dense_exact"
    count_bonus_exponent: float = 0.5
    count_min_count: float = 1.0
    count_ignore_empty_room_distractor: bool = False
    simhash_mode: SimHashMode = "static"
    simhash_bits: int = 32
    simhash_table_size: int = 16384
    simhash_table_overflow: CountTableOverflow = "warn"
    simhash_bonus_exponent: float = 0.5
    simhash_min_count: float = 1.0
    simhash_update_period: int = 1
    simhash_ignore_empty_room_distractor: bool = False


def dqn_agent_config(component_id: str, config: dict[str, Any]) -> DqnAgentConfig:
    if component_id not in {DQN_AGENT_COMPONENT, DQN_RMAX_AGENT_COMPONENT}:
        raise ValueError(
            f"Unsupported DQN agent {component_id!r}. Use {DQN_AGENT_COMPONENT} "
            f"or {DQN_RMAX_AGENT_COMPONENT} and connect R-Max knownness through "
            "the knownness_signal port."
        )
    algorithm: AgentAlgorithm = "dqn_rmax" if component_id == DQN_RMAX_AGENT_COMPONENT else "dqn"
    return DqnAgentConfig(
        algorithm=algorithm,
        learning_rate=float(config["learning_rate"]),
        discount=float(config["discount"]),
        hidden_units=_hidden_units(config, "hidden"),
        activation=str(config.get("activation", "relu")),
        update_frequency=int(config["update_frequency"]),
        target_update_frequency=int(
            config.get("target_update_freq", config["target_update_frequency"])
        ),
        epsilon_start=float(config.get("eps_start", config["epsilon_start"])),
        epsilon_end=float(config.get("eps_end", config["epsilon_end"])),
        epsilon_decay_steps=int(config.get("eps_decay_steps", config["epsilon_decay_steps"])),
        eval_epsilon=float(config["eval_epsilon"]),
        loss_type=str(config["loss_type"]),
        huber_delta=float(config.get("huber_delta", 1.0)),
        double_q=bool(config.get("double_q", False)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        optimizer=str(config.get("optimizer", "adam")),
        optimizer_beta1=float(config.get("optimizer_beta1", 0.9)),
        optimizer_beta2=float(config.get("optimizer_beta2", 0.999)),
        optimizer_epsilon=float(config.get("optimizer_epsilon", 1e-8)),
        optimizer_weight_decay=float(config.get("optimizer_weight_decay", 0.0)),
        optimizer_momentum=float(config.get("optimizer_momentum", 0.0)),
        optimizer_decay=float(config.get("optimizer_decay", 0.95)),
        optimizer_centered=bool(config.get("optimizer_centered", False)),
        normalize_observations=bool(config.get("normalize_observations", False)),
        obs_normalization_epsilon=float(config.get("obs_normalization_epsilon", 1e-8)),
        obs_normalization_clip=config.get("obs_normalization_clip", 5.0),
        rmax_bonus_threshold=float(config.get("rmax_bonus_threshold", 0.5)),
        rmax_decision_v_max=float(
            config.get(
                "rmax_decision_v_max",
                config.get("rmax_v_max", 1.0 / max(1.0 - float(config["discount"]), 1e-6)),
            )
        ),
        rmax_update_v_max=float(
            config.get(
                "rmax_update_v_max",
                config.get("rmax_v_max", 1.0 / max(1.0 - float(config["discount"]), 1e-6)),
            )
        ),
        seed=int(config.get("seed", 0)),
    )


def dqn_replay_config(component_id: str, config: dict[str, Any]) -> DqnReplayConfig:
    if component_id not in DQN_REPLAY_COMPONENTS:
        raise ValueError(
            "DQN requires builtin.replay.uniform on the runner replay_buffer port, "
            f"got {component_id!r}."
        )
    capacity = int(config["capacity"])
    batch_size = int(config["batch_size"])
    min_size = int(config["min_size"])
    if min_size > capacity:
        raise ValueError(f"{component_id} min_size cannot exceed capacity")
    if batch_size > capacity:
        raise ValueError(f"{component_id} batch_size cannot exceed capacity")
    updates_per_step = int(config["updates_per_step"])
    intrinsic_updates_per_step = _optional_update_count(
        config,
        "intrinsic_updates_per_step",
        updates_per_step,
    )
    q_network_updates_per_step = _optional_update_count(
        config,
        "q_network_updates_per_step",
        updates_per_step,
    )
    if intrinsic_updates_per_step < 0:
        raise ValueError(f"{component_id} intrinsic_updates_per_step cannot be negative")
    if q_network_updates_per_step < 0:
        raise ValueError(f"{component_id} q_network_updates_per_step cannot be negative")
    return DqnReplayConfig(
        name=component_id,
        capacity=capacity,
        batch_size=batch_size,
        min_size=min_size,
        updates_per_step=updates_per_step,
        intrinsic_updates_per_step=intrinsic_updates_per_step,
        q_network_updates_per_step=q_network_updates_per_step,
        save_dataset_path=str(config.get("save_dataset_path", "")),
    )


def _optional_update_count(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    return default if value is None else int(value)


def dqn_intrinsic_config(
    component_id: str | None,
    config: dict[str, Any] | None,
    agent: DqnAgentConfig,
) -> DqnIntrinsicConfig:
    if component_id is None:
        return DqnIntrinsicConfig(
            hidden_units=agent.hidden_units,
            activation=agent.activation,
            optimizer=agent.optimizer,
            learning_rate=agent.learning_rate,
        )
    config = config or {}
    if component_id == "builtin.intrinsic.rnd":
        return DqnIntrinsicConfig(
            kind="rnd",
            intrinsic_reward_scale=float(config["intrinsic_reward_scale"]),
            intrinsic_stats_decay=float(config["intrinsic_stats_decay"]),
            intrinsic_reward_epsilon=float(config["intrinsic_reward_epsilon"]),
            intrinsic_reward_clip=config["intrinsic_reward_clip"],
            intrinsic_reward_center=bool(config["intrinsic_reward_center"]),
            hidden_units=_hidden_units(config, "rnd", default=agent.hidden_units),
            activation=str(config.get("rnd_activation") or agent.activation),
            output_dim=int(config["rnd_output_dim"]),
            optimizer=str(config.get("rnd_optimizer") or agent.optimizer),
            learning_rate=float(config.get("rnd_learning_rate") or agent.learning_rate),
            action_conditioning=_resolve_action_conditioning(
                config.get("rnd_include_action"),
                config["rnd_action_conditioning"],
            ),
            update_period=int(config["rnd_update_period"]),
        )
    if component_id == "builtin.intrinsic.cfn":
        return DqnIntrinsicConfig(
            kind="cfn",
            intrinsic_reward_scale=float(config["intrinsic_reward_scale"]),
            intrinsic_stats_decay=float(config["intrinsic_stats_decay"]),
            intrinsic_reward_epsilon=float(config["intrinsic_reward_epsilon"]),
            intrinsic_reward_clip=config["intrinsic_reward_clip"],
            intrinsic_reward_center=bool(config["intrinsic_reward_center"]),
            hidden_units=_hidden_units(config, "cfn", default=agent.hidden_units),
            activation=str(config.get("cfn_activation") or agent.activation),
            output_dim=int(config["cfn_output_dim"]),
            optimizer=str(config.get("cfn_optimizer") or agent.optimizer),
            learning_rate=float(config.get("cfn_learning_rate") or agent.learning_rate),
            action_conditioning=_canonicalize_action_conditioning(
                config["cfn_action_conditioning"]
            ),
            update_period=int(config["cfn_update_period"]),
            cfn_use_random_prior=bool(config["cfn_use_random_prior"]),
            cfn_prior_scale=float(config["cfn_prior_scale"]),
            cfn_bonus_exponent=float(config["cfn_bonus_exponent"]),
            cfn_final_tanh=bool(config["cfn_final_tanh"]),
        )
    if component_id == "builtin.intrinsic.count":
        return DqnIntrinsicConfig(
            kind="count",
            intrinsic_reward_scale=float(config["intrinsic_reward_scale"]),
            intrinsic_stats_decay=float(config["intrinsic_stats_decay"]),
            intrinsic_reward_epsilon=float(config["intrinsic_reward_epsilon"]),
            intrinsic_reward_clip=config["intrinsic_reward_clip"],
            intrinsic_reward_center=bool(config["intrinsic_reward_center"]),
            hidden_units=(),
            activation=agent.activation,
            output_dim=1,
            optimizer=agent.optimizer,
            learning_rate=agent.learning_rate,
            action_conditioning=_canonicalize_action_conditioning(
                config["count_action_conditioning"]
            ),
            count_table_size=int(config["count_table_size"]),
            count_table_overflow=_count_table_overflow_mode(
                config.get("count_table_overflow", "warn")
            ),
            count_key_mode=_count_key_mode(config.get("count_key_mode", "dense_exact")),
            count_bonus_exponent=float(config["count_bonus_exponent"]),
            count_min_count=float(config["count_min_count"]),
            count_ignore_empty_room_distractor=bool(
                config.get("count_ignore_empty_room_distractor", False)
            ),
        )
    if component_id == "builtin.intrinsic.simhash":
        return DqnIntrinsicConfig(
            kind="simhash",
            intrinsic_reward_scale=float(config["intrinsic_reward_scale"]),
            intrinsic_stats_decay=float(config["intrinsic_stats_decay"]),
            intrinsic_reward_epsilon=float(config["intrinsic_reward_epsilon"]),
            intrinsic_reward_clip=config["intrinsic_reward_clip"],
            intrinsic_reward_center=bool(config["intrinsic_reward_center"]),
            hidden_units=_hidden_units(config, "simhash", default=agent.hidden_units),
            activation=str(config.get("simhash_activation") or agent.activation),
            output_dim=int(config["simhash_latent_dim"]),
            optimizer=str(config.get("simhash_optimizer") or agent.optimizer),
            learning_rate=float(config.get("simhash_learning_rate") or agent.learning_rate),
            action_conditioning=_canonicalize_action_conditioning(
                config["simhash_action_conditioning"]
            ),
            update_period=int(config["simhash_update_period"]),
            simhash_mode=_simhash_mode(config["simhash_mode"]),
            simhash_bits=int(config["simhash_bits"]),
            simhash_table_size=int(config["simhash_table_size"]),
            simhash_table_overflow=_count_table_overflow_mode(
                config.get("simhash_table_overflow", "warn")
            ),
            simhash_bonus_exponent=float(config["simhash_bonus_exponent"]),
            simhash_min_count=float(config["simhash_min_count"]),
            simhash_update_period=int(config["simhash_update_period"]),
            simhash_ignore_empty_room_distractor=bool(
                config.get("simhash_ignore_empty_room_distractor", False)
            ),
        )
    raise ValueError(f"Unsupported DQN intrinsic reward component: {component_id}")


def _hidden_units(
    config: dict[str, Any],
    prefix: str,
    *,
    default: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    hidden_dims = config.get(f"{prefix}_hidden_dims")
    if hidden_dims is None and prefix == "hidden":
        hidden_dims = config.get("hidden_dims")
    if hidden_dims:
        return tuple(int(dim) for dim in hidden_dims)

    hidden_units = config.get(f"{prefix}_hidden_units")
    if hidden_units is None and prefix == "hidden":
        hidden_units = config.get("hidden_units")
    if hidden_units is None:
        return default or ()
    if isinstance(hidden_units, int):
        return (int(hidden_units),)
    if isinstance(hidden_units, str):
        return tuple(int(dim.strip()) for dim in hidden_units.split(",") if dim.strip())
    return tuple(int(dim) for dim in hidden_units)


def _canonicalize_action_conditioning(mode: str | bool) -> ActionConditioning:
    if isinstance(mode, bool):
        return "input" if mode else "none"
    normalized = str(mode).strip().lower()
    aliases = {
        "none": "none",
        "state": "none",
        "observation": "none",
        "input": "input",
        "action_input": "input",
        "include_action": "input",
        "output": "output",
        "action_output": "output",
        "per_action": "output",
        "pair": "pair",
        "state_action": "pair",
        "state_action_pair": "pair",
        "onehot_pair": "pair",
        "obs_action_onehot": "pair",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported action conditioning mode: {mode!r}")
    return aliases[normalized]  # type: ignore[return-value]


def _resolve_action_conditioning(
    legacy_include_action: bool | None,
    mode: str | bool,
) -> ActionConditioning:
    canonical = _canonicalize_action_conditioning(mode)
    if legacy_include_action is None:
        return canonical
    legacy = _canonicalize_action_conditioning(legacy_include_action)
    if canonical != "none" and canonical != legacy:
        raise ValueError(
            "rnd_include_action and rnd_action_conditioning disagree: "
            f"{legacy_include_action!r} vs {mode!r}"
        )
    return legacy


def _count_table_overflow_mode(mode: str) -> CountTableOverflow:
    normalized = str(mode).strip().lower()
    if normalized not in {"warn", "error"}:
        raise ValueError("count_table_overflow must be 'warn' or 'error'")
    return normalized  # type: ignore[return-value]


def _count_key_mode(value: str) -> CountKeyMode:
    normalized = str(value).strip().lower()
    if normalized not in {"dense_exact", "oracle_tabular"}:
        raise ValueError("count_key_mode must be 'dense_exact' or 'oracle_tabular'")
    return normalized  # type: ignore[return-value]


def _simhash_mode(mode: str) -> SimHashMode:
    normalized = str(mode).strip().lower()
    if normalized == "autoencoder":
        normalized = "learned"
    if normalized not in {"static", "learned"}:
        raise ValueError("simhash_mode must be 'static' or 'learned'")
    return normalized  # type: ignore[return-value]
