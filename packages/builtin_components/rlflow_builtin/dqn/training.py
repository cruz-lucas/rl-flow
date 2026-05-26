from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlflow_builtin.tabular.environments import environment_config, initial_state, make_step_fn
from rlflow_builtin.tabular.types import RunnerConfig


DQN_AGENT_COMPONENT = "builtin.agent.dqn_jax"
DQN_RMAX_AGENT_COMPONENT = "builtin.agent.dqn_rmax_jax"
DQN_REPLAY_COMPONENTS = {"builtin.replay.uniform", "builtin.replay.dqn_uniform"}

AgentAlgorithm = Literal["dqn", "dqn_rmax"]
IntrinsicKind = Literal["none", "rnd", "cfn", "count"]
ActionConditioning = Literal["none", "input", "output", "pair"]
CountTableOverflow = Literal["warn", "error"]


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
    rmax_v_max: float
    seed: int


@dataclass(frozen=True)
class DqnReplayConfig:
    name: str
    capacity: int
    batch_size: int
    min_size: int
    updates_per_step: int
    save_dataset_path: str = ""


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
    count_bonus_exponent: float = 0.5
    count_min_count: float = 1.0


@dataclass(frozen=True)
class DqnRunResult:
    params: tuple[dict[str, jax.Array], ...]
    aux_params: dict[str, tuple[dict[str, jax.Array], ...]]
    train_returns: np.ndarray
    train_lengths: np.ndarray
    train_losses: np.ndarray
    eval_returns: np.ndarray
    eval_lengths: np.ndarray
    source_observation_shape: tuple[int, ...]
    source_observation_dtype: str
    input_dim: int
    num_actions: int
    replay_arrays: dict[str, np.ndarray] | None = None
    count_table_entries: int | None = None
    count_table_overflow: bool | None = None


class DqnReplayState(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    intrinsic_targets: jax.Array
    source_observations: jax.Array
    source_next_observations: jax.Array
    size: jax.Array
    index: jax.Array


class DqnIntrinsicState(NamedTuple):
    target_params: tuple[dict[str, jax.Array], ...]
    prior_params: tuple[dict[str, jax.Array], ...]
    predictor_params: tuple[dict[str, jax.Array], ...]
    opt_state: optax.OptState
    reward_mean: jax.Array
    reward_var: jax.Array
    count_keys: jax.Array
    counts: jax.Array
    count_size: jax.Array
    count_overflow: jax.Array


class DqnTrainState(NamedTuple):
    params: tuple[dict[str, jax.Array], ...]
    target_params: tuple[dict[str, jax.Array], ...]
    opt_state: optax.OptState
    intrinsic_state: DqnIntrinsicState
    replay_state: DqnReplayState
    key: jax.Array
    global_step: jax.Array
    gradient_step: jax.Array


@dataclass(frozen=True)
class _DqnEnvironment:
    observation_shape: tuple[int, ...]
    observation_dtype: str
    input_dim: int
    num_actions: int
    reset: Callable[[jax.Array], Any]
    step: Callable[[Any, jax.Array, jax.Array], Any]
    observation: Callable[[Any], Any]
    reward: Callable[[Any], jax.Array]
    done: Callable[[Any], jax.Array]
    encode: Callable[[Any], jax.Array]


def dqn_agent_config(component_id: str, config: dict[str, Any]) -> DqnAgentConfig:
    if component_id not in {DQN_AGENT_COMPONENT, DQN_RMAX_AGENT_COMPONENT}:
        raise ValueError(
            f"Unsupported DQN agent {component_id!r}. Use {DQN_AGENT_COMPONENT} "
            f"or {DQN_RMAX_AGENT_COMPONENT} and connect intrinsic bonuses through "
            "the intrinsic_reward port."
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
        rmax_v_max=float(config.get("rmax_v_max", 1.0 / max(1.0 - float(config["discount"]), 1e-6))),
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
    return DqnReplayConfig(
        name=component_id,
        capacity=capacity,
        batch_size=batch_size,
        min_size=min_size,
        updates_per_step=int(config["updates_per_step"]),
        save_dataset_path=str(config.get("save_dataset_path", "")),
    )


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
            count_bonus_exponent=float(config["count_bonus_exponent"]),
            count_min_count=float(config["count_min_count"]),
        )
    raise ValueError(f"Unsupported DQN intrinsic reward component: {component_id}")


def run_dqn_training(
    *,
    env_component: str,
    env_settings: dict[str, Any],
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    runner: RunnerConfig,
    intrinsic: DqnIntrinsicConfig | None = None,
    run_dir: Path | None = None,
) -> DqnRunResult:
    intrinsic = intrinsic or dqn_intrinsic_config(None, None, agent)
    if agent.algorithm == "dqn_rmax" and intrinsic.kind == "none":
        raise ValueError("builtin.agent.dqn_rmax_jax requires an intrinsic_reward input")
    dqn_env = _make_dqn_environment(
        env_component,
        env_settings,
        normalize_observations=agent.normalize_observations,
    )
    seed = runner.seed + agent.seed
    key = jax.random.PRNGKey(seed)
    key, q_key, intrinsic_key = jax.random.split(key, 3)

    q_optimizer = _optimizer(agent, agent.learning_rate, agent.optimizer)
    q_params = _init_mlp(q_key, dqn_env.input_dim, agent.hidden_units, dqn_env.num_actions)
    initial_state = DqnTrainState(
        params=q_params,
        target_params=_clone_params(q_params),
        opt_state=q_optimizer.init(q_params),
        intrinsic_state=_initial_intrinsic_state(
            agent,
            intrinsic,
            dqn_env.input_dim,
            dqn_env.num_actions,
            intrinsic_key,
        ),
        replay_state=_initial_replay_state(
            replay.capacity,
            dqn_env.input_dim,
            _replay_intrinsic_target_dim(intrinsic),
            dqn_env.observation_shape,
            dqn_env.observation_dtype,
        ),
        key=key,
        global_step=jnp.asarray(0, dtype=jnp.int32),
        gradient_step=jnp.asarray(0, dtype=jnp.int32),
    )
    intrinsic_optimizer = _optimizer(agent, intrinsic.learning_rate, intrinsic.optimizer)

    if runner.train_steps is None:

        @jax.jit
        def train_scan(state: DqnTrainState):
            return jax.lax.scan(
                lambda carry, _: _train_episode(
                    carry,
                    dqn_env,
                    agent,
                    replay,
                    intrinsic,
                    q_optimizer,
                    intrinsic_optimizer,
                    runner.max_episode_steps,
                ),
                state,
                xs=None,
                length=runner.train_episodes,
            )

        final_state, train_history = train_scan(initial_state)
    else:

        @jax.jit
        def train_scan(state: DqnTrainState):
            return _train_steps(
                state,
                dqn_env,
                agent,
                replay,
                intrinsic,
                q_optimizer,
                intrinsic_optimizer,
                runner.max_episode_steps,
                runner.train_steps or 0,
            )

        final_state, train_history = train_scan(initial_state)

    count_entries, count_overflow = _count_table_status(final_state.intrinsic_state, intrinsic)
    _handle_count_table_overflow(intrinsic, count_overflow)

    if runner.eval_episodes > 0:

        @jax.jit
        def eval_scan(eval_key: jax.Array):
            return jax.lax.scan(
                lambda carry, _: _eval_episode(
                    carry,
                    final_state.params,
                    final_state.intrinsic_state,
                    dqn_env,
                    agent,
                    intrinsic,
                    runner.max_episode_steps,
                ),
                eval_key,
                xs=None,
                length=runner.eval_episodes,
            )

        _, eval_history = eval_scan(jax.random.PRNGKey(seed + 10000))
        eval_returns, eval_lengths = eval_history
    else:
        eval_returns = jnp.asarray([], dtype=jnp.float32)
        eval_lengths = jnp.asarray([], dtype=jnp.int32)

    train_returns, train_lengths, train_losses = train_history
    train_returns_np = np.asarray(train_returns)
    train_lengths_np = np.asarray(train_lengths)
    train_losses_np = np.asarray(train_losses)
    if runner.train_steps is not None:
        episode_count = int(np.count_nonzero(train_lengths_np))
        train_returns_np = train_returns_np[:episode_count]
        train_lengths_np = train_lengths_np[:episode_count]
        train_losses_np = train_losses_np[:episode_count]
    replay_arrays = (
        _replay_arrays(final_state.replay_state, intrinsic)
        if replay.save_dataset_path
        else None
    )
    if replay.save_dataset_path and run_dir is not None and replay_arrays is not None:
        save_path = _resolve_replay_save_path(replay.save_dataset_path, run_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **replay_arrays)

    return DqnRunResult(
        params=final_state.params,
        aux_params=_aux_params(final_state.intrinsic_state, intrinsic),
        train_returns=train_returns_np,
        train_lengths=train_lengths_np,
        train_losses=train_losses_np,
        eval_returns=np.asarray(eval_returns),
        eval_lengths=np.asarray(eval_lengths),
        source_observation_shape=dqn_env.observation_shape,
        source_observation_dtype=dqn_env.observation_dtype,
        input_dim=dqn_env.input_dim,
        num_actions=dqn_env.num_actions,
        replay_arrays=replay_arrays,
        count_table_entries=count_entries,
        count_table_overflow=count_overflow,
    )


def _train_episode(
    state: DqnTrainState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    intrinsic: DqnIntrinsicConfig,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    max_episode_steps: int,
) -> tuple[DqnTrainState, tuple[jax.Array, jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(state.key)
    env_state = dqn_env.reset(reset_key)
    source_observation = dqn_env.observation(env_state)
    observation = dqn_env.encode(source_observation)
    step_carry = (
        state._replace(key=key),
        env_state,
        observation,
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def step_fn(carry, _):
        train_state, env_state, observation, done, episode_return, episode_length, loss_sum = carry

        def inactive(active_carry):
            return active_carry, jnp.asarray(0.0, dtype=jnp.float32)

        def active(active_carry):
            train_state, env_state, observation, _done, episode_return, episode_length, loss_sum = active_carry
            key, action_key, env_key, replay_key, update_key = jax.random.split(
                train_state.key,
                5,
            )
            action = _select_action(
                agent,
                train_state.params,
                train_state.intrinsic_state,
                intrinsic,
                observation,
                action_key,
                dqn_env.num_actions,
                train_state.global_step,
                training=True,
            )
            next_env_state = dqn_env.step(env_state, action, env_key)
            next_source_observation = dqn_env.observation(next_env_state)
            next_observation = dqn_env.encode(next_source_observation)
            reward = dqn_env.reward(next_env_state).astype(jnp.float32)
            terminal = dqn_env.done(next_env_state)
            intrinsic_target = _sample_intrinsic_target(
                intrinsic,
                replay_key,
                train_state.replay_state.intrinsic_targets.shape[-1],
            )
            replay_state = _push_replay(
                train_state.replay_state,
                observation,
                dqn_env.observation(env_state),
                action,
                reward,
                next_observation,
                next_source_observation,
                terminal,
                intrinsic_target,
            )
            intrinsic_state = _observe_intrinsic_transition(
                train_state.intrinsic_state,
                observation,
                action,
                intrinsic,
                dqn_env.num_actions,
            )
            train_state = train_state._replace(
                replay_state=replay_state,
                intrinsic_state=intrinsic_state,
                key=update_key,
            )
            should_update = jnp.logical_and(
                replay_state.size >= replay.min_size,
                train_state.global_step % agent.update_frequency == 0,
            )
            train_state, step_loss = jax.lax.cond(
                should_update,
                lambda update_state: _replay_updates(
                    update_state,
                    agent,
                    replay,
                    intrinsic,
                    q_optimizer,
                    intrinsic_optimizer,
                    dqn_env.num_actions,
                ),
                lambda update_state: (
                    update_state,
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                train_state,
            )
            train_state = train_state._replace(
                global_step=train_state.global_step + 1,
            )
            return (
                train_state,
                next_env_state,
                next_observation,
                terminal,
                episode_return + reward,
                episode_length + 1,
                loss_sum + step_loss,
            ), step_loss

        return jax.lax.cond(done, inactive, active, carry)

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=max_episode_steps,
    )
    state, _, _, _, episode_return, episode_length, loss_sum = step_carry
    mean_loss = loss_sum / jnp.maximum(episode_length, 1)
    return state, (episode_return, episode_length, mean_loss)


def _train_steps(
    state: DqnTrainState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    intrinsic: DqnIntrinsicConfig,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    max_episode_steps: int,
    train_steps: int,
) -> tuple[DqnTrainState, tuple[jax.Array, jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(state.key)
    env_state = dqn_env.reset(reset_key)
    source_observation = dqn_env.observation(env_state)
    observation = dqn_env.encode(source_observation)
    train_returns = jnp.zeros((train_steps,), dtype=jnp.float32)
    train_lengths = jnp.zeros((train_steps,), dtype=jnp.int32)
    train_losses = jnp.zeros((train_steps,), dtype=jnp.float32)
    step_carry = (
        state._replace(key=key),
        env_state,
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        train_returns,
        train_lengths,
        train_losses,
    )

    def write_episode(
        episode_index: jax.Array,
        returns: jax.Array,
        lengths: jax.Array,
        losses: jax.Array,
        episode_return: jax.Array,
        episode_length: jax.Array,
        loss_sum: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        mean_loss = loss_sum / jnp.maximum(episode_length, 1)
        return (
            episode_index + 1,
            returns.at[episode_index].set(episode_return),
            lengths.at[episode_index].set(episode_length),
            losses.at[episode_index].set(mean_loss),
        )

    def step_fn(carry, _):
        (
            train_state,
            env_state,
            observation,
            episode_return,
            episode_length,
            loss_sum,
            episode_index,
            returns,
            lengths,
            losses,
        ) = carry
        _key, action_key, env_key, replay_key, update_key, reset_key = jax.random.split(
            train_state.key,
            6,
        )
        action = _select_action(
            agent,
            train_state.params,
            train_state.intrinsic_state,
            intrinsic,
            observation,
            action_key,
            dqn_env.num_actions,
            train_state.global_step,
            training=True,
        )
        next_env_state = dqn_env.step(env_state, action, env_key)
        next_source_observation = dqn_env.observation(next_env_state)
        next_observation = dqn_env.encode(next_source_observation)
        reward = dqn_env.reward(next_env_state).astype(jnp.float32)
        terminal = dqn_env.done(next_env_state)
        intrinsic_target = _sample_intrinsic_target(
            intrinsic,
            replay_key,
            train_state.replay_state.intrinsic_targets.shape[-1],
        )
        replay_state = _push_replay(
            train_state.replay_state,
            observation,
            dqn_env.observation(env_state),
            action,
            reward,
            next_observation,
            next_source_observation,
            terminal,
            intrinsic_target,
        )
        intrinsic_state = _observe_intrinsic_transition(
            train_state.intrinsic_state,
            observation,
            action,
            intrinsic,
            dqn_env.num_actions,
        )
        train_state = train_state._replace(
            replay_state=replay_state,
            intrinsic_state=intrinsic_state,
            key=update_key,
        )
        should_update = jnp.logical_and(
            replay_state.size >= replay.min_size,
            train_state.global_step % agent.update_frequency == 0,
        )
        train_state, step_loss = jax.lax.cond(
            should_update,
            lambda update_state: _replay_updates(
                update_state,
                agent,
                replay,
                intrinsic,
                q_optimizer,
                intrinsic_optimizer,
                dqn_env.num_actions,
            ),
            lambda update_state: (
                update_state,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            train_state,
        )
        train_state = train_state._replace(global_step=train_state.global_step + 1)
        next_episode_return = episode_return + reward
        next_episode_length = episode_length + 1
        next_loss_sum = loss_sum + step_loss
        episode_done = jnp.logical_or(terminal, next_episode_length >= max_episode_steps)

        def finish_episode(args):
            (
                train_state,
                _next_env_state,
                _next_observation,
                next_episode_return,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                lengths,
                losses,
                reset_key,
            ) = args
            episode_index, returns, lengths, losses = write_episode(
                episode_index,
                returns,
                lengths,
                losses,
                next_episode_return,
                next_episode_length,
                next_loss_sum,
            )
            reset_env_state = dqn_env.reset(reset_key)
            reset_source_observation = dqn_env.observation(reset_env_state)
            reset_observation = dqn_env.encode(reset_source_observation)
            return (
                train_state,
                reset_env_state,
                reset_observation,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
                episode_index,
                returns,
                lengths,
                losses,
            )

        def continue_episode(args):
            (
                train_state,
                next_env_state,
                next_observation,
                next_episode_return,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                lengths,
                losses,
                _reset_key,
            ) = args
            return (
                train_state,
                next_env_state,
                next_observation,
                next_episode_return,
                next_episode_length,
                next_loss_sum,
                episode_index,
                returns,
                lengths,
                losses,
            )

        return (
            jax.lax.cond(
                episode_done,
                finish_episode,
                continue_episode,
                (
                    train_state,
                    next_env_state,
                    next_observation,
                    next_episode_return,
                    next_episode_length,
                    next_loss_sum,
                    episode_index,
                    returns,
                    lengths,
                    losses,
                    reset_key,
                ),
            ),
            None,
        )

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=train_steps,
    )
    (
        state,
        _env_state,
        _observation,
        episode_return,
        episode_length,
        loss_sum,
        episode_index,
        returns,
        lengths,
        losses,
    ) = step_carry

    def write_partial(args):
        episode_index, returns, lengths, losses = write_episode(
            args[0],
            args[1],
            args[2],
            args[3],
            episode_return,
            episode_length,
            loss_sum,
        )
        return episode_index, returns, lengths, losses

    def skip_partial(args):
        return args

    _episode_index, returns, lengths, losses = jax.lax.cond(
        episode_length > 0,
        write_partial,
        skip_partial,
        (episode_index, returns, lengths, losses),
    )
    return state, (returns, lengths, losses)


def _replay_updates(
    state: DqnTrainState,
    agent: DqnAgentConfig,
    replay: DqnReplayConfig,
    intrinsic: DqnIntrinsicConfig,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
) -> tuple[DqnTrainState, jax.Array]:
    def update_step(carry, _):
        train_state, loss_sum = carry
        key, sample_key = jax.random.split(train_state.key)
        batch = _sample_replay(train_state.replay_state, sample_key, replay.batch_size)
        train_state = train_state._replace(key=key)
        train_state, loss = _update_from_batch(
            train_state,
            batch,
            agent,
            intrinsic,
            q_optimizer,
            intrinsic_optimizer,
            num_actions,
        )
        return (train_state, loss_sum + loss), loss

    (state, loss_sum), _ = jax.lax.scan(
        update_step,
        (state, jnp.asarray(0.0, dtype=jnp.float32)),
        xs=None,
        length=replay.updates_per_step,
    )
    return state, loss_sum / replay.updates_per_step


def _update_from_batch(
    state: DqnTrainState,
    batch: dict[str, jax.Array],
    agent: DqnAgentConfig,
    intrinsic: DqnIntrinsicConfig,
    q_optimizer: optax.GradientTransformation,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
) -> tuple[DqnTrainState, jax.Array]:
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_observations = batch["next_observations"]
    terminals = batch["terminals"]
    rmax_known_mask, rmax_next_unknown = _rmax_batch_masks(
        state.intrinsic_state,
        intrinsic,
        agent,
        observations,
        actions,
        next_observations,
        num_actions,
    )
    total_rewards, intrinsic_state, intrinsic_loss = _intrinsic_update(
        state.intrinsic_state,
        batch,
        intrinsic,
        intrinsic_optimizer,
        num_actions,
        state.gradient_step + 1,
    )
    intrinsic_reward_scale = (
        0.0 if agent.algorithm == "dqn_rmax" else intrinsic.intrinsic_reward_scale
    )
    total_rewards = rewards + intrinsic_reward_scale * jax.lax.stop_gradient(total_rewards)

    def q_loss_fn(params):
        return _q_loss(
            agent,
            params,
            state.target_params,
            observations,
            actions,
            total_rewards,
            next_observations,
            terminals,
            rmax_known_mask,
            rmax_next_unknown,
        )

    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(state.params)
    q_updates, opt_state = q_optimizer.update(q_grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, q_updates)
    gradient_step = state.gradient_step + 1
    should_update_target = gradient_step % agent.target_update_frequency == 0
    target_params = _maybe_hard_update(params, state.target_params, should_update_target)
    return (
        state._replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            intrinsic_state=intrinsic_state,
            gradient_step=gradient_step,
        ),
        q_loss + intrinsic_loss,
    )


def _intrinsic_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    if intrinsic.kind == "none":
        return (
            jnp.zeros_like(batch["rewards"]),
            state,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    if intrinsic.kind == "rnd":
        return _rnd_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
    if intrinsic.kind == "cfn":
        return _cfn_update(
            state,
            batch,
            intrinsic,
            intrinsic_optimizer,
            num_actions,
            next_gradient_step,
        )
    return _count_update(
        state,
        batch,
        intrinsic,
        num_actions,
    )


def _rnd_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    actions = batch["actions"]
    # TODO: add next state or current state option
    intrinsic_observations = (
        # batch["next_observations"]
        # if intrinsic.action_conditioning == "none"
        # else batch["observations"]
        batch["observations"]
    )
    prediction_error, intrinsic_input, target_features = _rnd_prediction_error(
        state.target_params,
        state.predictor_params,
        intrinsic_observations,
        actions,
        intrinsic,
        num_actions,
    )
    intrinsic_reward = _normalize_intrinsic_reward(
        intrinsic,
        prediction_error,
        state.reward_mean,
        state.reward_var,
    )

    def loss_fn(predictor_params):
        predictor_features = _select_conditioned_features(
            _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
        return jnp.mean(jnp.square(predictor_features - target_features))

    def do_predictor_update(args):
        predictor_params, opt_state = args
        intrinsic_loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = intrinsic_optimizer.update(
            grads,
            opt_state,
            predictor_params,
        )
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, intrinsic_loss

    def skip_predictor_update(args):
        predictor_params, opt_state = args
        return predictor_params, opt_state, loss_fn(predictor_params)

    predictor_params, opt_state, intrinsic_loss = jax.lax.cond(
        next_gradient_step % intrinsic.update_period == 0,
        do_predictor_update,
        skip_predictor_update,
        (state.predictor_params, state.opt_state),
    )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(prediction_error),
    )
    return (
        intrinsic_reward,
        state._replace(
            predictor_params=predictor_params,
            opt_state=opt_state,
            reward_mean=reward_mean,
            reward_var=reward_var,
        ),
        intrinsic_loss,
    )


def _cfn_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    intrinsic_optimizer: optax.GradientTransformation,
    num_actions: int,
    next_gradient_step: jax.Array,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    actions = batch["actions"]
    # TODO: add next state or current state option
    intrinsic_observations = (
        # batch["next_observations"]
        # if intrinsic.action_conditioning == "none"
        # else batch["observations"]
        batch["observations"]
    )
    raw_bonus, intrinsic_input, prior_features, _predictor_features, _coin_flips = _cfn_outputs(
        state.prior_params,
        state.predictor_params,
        intrinsic_observations,
        actions,
        intrinsic,
        num_actions,
    )
    intrinsic_reward = _normalize_intrinsic_reward(
        intrinsic,
        raw_bonus,
        state.reward_mean,
        state.reward_var,
    )
    targets = batch["intrinsic_targets"]

    def loss_fn(predictor_params):
        predictor_features = _maybe_tanh(
            _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
            intrinsic.cfn_final_tanh,
        )
        predictor_features = _select_conditioned_features(
            predictor_features,
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
        if intrinsic.cfn_use_random_prior:
            coin_flips = predictor_features + intrinsic.cfn_prior_scale * prior_features
        else:
            coin_flips = predictor_features
        return jnp.mean(jnp.square(coin_flips - targets))

    def do_predictor_update(args):
        predictor_params, opt_state = args
        intrinsic_loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = intrinsic_optimizer.update(
            grads,
            opt_state,
            predictor_params,
        )
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, intrinsic_loss

    def skip_predictor_update(args):
        predictor_params, opt_state = args
        return predictor_params, opt_state, loss_fn(predictor_params)

    predictor_params, opt_state, intrinsic_loss = jax.lax.cond(
        next_gradient_step % intrinsic.update_period == 0,
        do_predictor_update,
        skip_predictor_update,
        (state.predictor_params, state.opt_state),
    )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(raw_bonus),
    )
    return (
        intrinsic_reward,
        state._replace(
            predictor_params=predictor_params,
            opt_state=opt_state,
            reward_mean=reward_mean,
            reward_var=reward_var,
        ),
        intrinsic_loss,
    )


def _count_update(
    state: DqnIntrinsicState,
    batch: dict[str, jax.Array],
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, DqnIntrinsicState, jax.Array]:
    raw_bonus = _count_raw_bonus(
        state,
        batch["observations"],
        batch["actions"],
        intrinsic,
        num_actions,
    )
    intrinsic_reward = raw_bonus
    # _normalize_intrinsic_reward(
    #     intrinsic,
    #     raw_bonus,
    #     state.reward_mean,
    #     state.reward_var,
    # )
    reward_mean, reward_var = _update_intrinsic_stats(
        intrinsic,
        state.reward_mean,
        state.reward_var,
        jax.lax.stop_gradient(raw_bonus),
    )
    return (
        intrinsic_reward,
        state._replace(reward_mean=reward_mean, reward_var=reward_var),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _eval_episode(
    key: jax.Array,
    params: tuple[dict[str, jax.Array], ...],
    intrinsic_state: DqnIntrinsicState,
    dqn_env: _DqnEnvironment,
    agent: DqnAgentConfig,
    intrinsic: DqnIntrinsicConfig,
    max_episode_steps: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    key, reset_key = jax.random.split(key)
    env_state = dqn_env.reset(reset_key)
    observation = dqn_env.encode(dqn_env.observation(env_state))
    step_carry = (
        key,
        env_state,
        observation,
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step_fn(carry, _):
        key, env_state, observation, done, episode_return, episode_length = carry

        def inactive(active_carry):
            return active_carry, None

        def active(active_carry):
            key, env_state, observation, _done, episode_return, episode_length = active_carry
            key, action_key, env_key = jax.random.split(key, 3)
            action = _select_action(
                agent,
                params,
                intrinsic_state,
                intrinsic,
                observation,
                action_key,
                dqn_env.num_actions,
                jnp.asarray(0, dtype=jnp.int32),
                training=False,
            )
            next_env_state = dqn_env.step(env_state, action, env_key)
            reward = dqn_env.reward(next_env_state).astype(jnp.float32)
            terminal = dqn_env.done(next_env_state)
            next_observation = dqn_env.encode(dqn_env.observation(next_env_state))
            return (
                key,
                next_env_state,
                next_observation,
                terminal,
                episode_return + reward,
                episode_length + 1,
            ), None

        return jax.lax.cond(done, inactive, active, carry)

    step_carry, _ = jax.lax.scan(
        step_fn,
        step_carry,
        xs=None,
        length=max_episode_steps,
    )
    key, _, _, _, episode_return, episode_length = step_carry
    return key, (episode_return, episode_length)


def _make_dqn_environment(
    env_component: str,
    env_settings: dict[str, Any],
    *,
    normalize_observations: bool = False,
) -> _DqnEnvironment:
    if env_component == "navix.env.grid":
        from rlflow_builtin.environments.navix import create_navix_environment

        env = create_navix_environment(**_coerce_navix_settings(env_settings))
        shape = tuple(env.observation_space.shape)
        input_dim = _input_dim_from_space(env.observation_space)
        observation_dtype = np.dtype(env.observation_space.dtype)
        is_scalar = shape in {(), (1,)}
        is_integer = np.issubdtype(observation_dtype, np.integer)
        normalization_scale = _integer_observation_scale(observation_dtype)

        def encode(observation):
            observation = jnp.asarray(observation)
            if is_scalar:
                return jax.nn.one_hot(
                    observation.reshape(()).astype(jnp.int32),
                    input_dim,
                    dtype=jnp.float32,
                )
            encoded = observation.astype(jnp.float32).reshape(-1)
            if normalize_observations and is_integer and normalization_scale is not None:
                encoded = encoded / normalization_scale
            return encoded

        return _DqnEnvironment(
            observation_shape=shape,
            observation_dtype=str(observation_dtype),
            input_dim=input_dim,
            num_actions=int(env.action_space.n),
            reset=lambda key: env.reset(key),
            step=lambda timestep, action, _key: env.step(timestep, action),
            observation=lambda timestep: timestep.observation,
            reward=lambda timestep: timestep.reward,
            done=_timestep_done,
            encode=encode,
        )

    tabular_env = environment_config(env_component, env_settings)
    step_fn = make_step_fn(tabular_env)

    def reset(key):
        return (
            initial_state(tabular_env, key),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
        )

    def step(state, action, key):
        current_state = state[0] if isinstance(state, tuple) else state
        return step_fn(current_state, action, key)

    def encode(observation):
        return jax.nn.one_hot(
            jnp.asarray(observation).reshape(()).astype(jnp.int32),
            tabular_env.num_states,
            dtype=jnp.float32,
        )

    return _DqnEnvironment(
        observation_shape=(),
        observation_dtype="int32",
        input_dim=tabular_env.num_states,
        num_actions=tabular_env.num_actions,
        reset=reset,
        step=step,
        observation=lambda state: state[0] if isinstance(state, tuple) else state,
        reward=lambda state: state[1] if isinstance(state, tuple) else jnp.asarray(0.0, dtype=jnp.float32),
        done=lambda state: state[2] if isinstance(state, tuple) else jnp.asarray(False),
        encode=encode,
    )


def _initial_replay_state(
    capacity: int,
    input_dim: int,
    intrinsic_target_dim: int,
    source_observation_shape: tuple[int, ...],
    source_observation_dtype: str,
) -> DqnReplayState:
    source_shape = (capacity, *source_observation_shape)
    source_dtype = np.dtype(source_observation_dtype)
    return DqnReplayState(
        observations=jnp.zeros((capacity, input_dim), dtype=jnp.float32),
        actions=jnp.zeros((capacity,), dtype=jnp.int32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_observations=jnp.zeros((capacity, input_dim), dtype=jnp.float32),
        terminals=jnp.zeros((capacity,), dtype=jnp.float32),
        intrinsic_targets=jnp.zeros((capacity, intrinsic_target_dim), dtype=jnp.float32),
        source_observations=jnp.zeros(source_shape, dtype=source_dtype),
        source_next_observations=jnp.zeros(source_shape, dtype=source_dtype),
        size=jnp.asarray(0, dtype=jnp.int32),
        index=jnp.asarray(0, dtype=jnp.int32),
    )


def _push_replay(
    state: DqnReplayState,
    observation: jax.Array,
    source_observation: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_observation: jax.Array,
    source_next_observation: jax.Array,
    terminal: jax.Array,
    intrinsic_target: jax.Array,
) -> DqnReplayState:
    index = state.index
    capacity = state.observations.shape[0]
    return DqnReplayState(
        observations=state.observations.at[index].set(observation),
        actions=state.actions.at[index].set(action.astype(jnp.int32)),
        rewards=state.rewards.at[index].set(reward.astype(jnp.float32)),
        next_observations=state.next_observations.at[index].set(next_observation),
        terminals=state.terminals.at[index].set(terminal.astype(jnp.float32)),
        intrinsic_targets=state.intrinsic_targets.at[index].set(intrinsic_target),
        source_observations=state.source_observations.at[index].set(
            source_observation.astype(state.source_observations.dtype)
        ),
        source_next_observations=state.source_next_observations.at[index].set(
            source_next_observation.astype(state.source_next_observations.dtype)
        ),
        size=jnp.minimum(state.size + 1, capacity).astype(jnp.int32),
        index=((index + 1) % capacity).astype(jnp.int32),
    )


def _sample_replay(
    state: DqnReplayState,
    key: jax.Array,
    batch_size: int,
) -> dict[str, jax.Array]:
    indices = jax.random.randint(key, (batch_size,), 0, state.size, dtype=jnp.int32)
    return {
        "observations": state.observations[indices],
        "actions": state.actions[indices],
        "rewards": state.rewards[indices],
        "next_observations": state.next_observations[indices],
        "terminals": state.terminals[indices],
        "intrinsic_targets": state.intrinsic_targets[indices],
    }


def _initial_intrinsic_state(
    agent: DqnAgentConfig,
    intrinsic: DqnIntrinsicConfig,
    input_dim: int,
    num_actions: int,
    key: jax.Array,
) -> DqnIntrinsicState:
    key, target_key, prior_key, predictor_key = jax.random.split(key, 4)
    del key
    if intrinsic.kind == "none":
        target_input_dim = 1
        target_output_dim = 1
        hidden_units: tuple[int, ...] = ()
    else:
        target_input_dim = _conditioned_input_dim(
            input_dim,
            num_actions,
            intrinsic.action_conditioning,
        )
        target_output_dim = _conditioned_output_dim(
            num_actions,
            intrinsic.output_dim,
            intrinsic.action_conditioning,
        )
        hidden_units = intrinsic.hidden_units
    target_params = _init_mlp(target_key, target_input_dim, hidden_units, target_output_dim)
    prior_params = _init_mlp(prior_key, target_input_dim, hidden_units, target_output_dim)
    predictor_params = _init_mlp(
        predictor_key,
        target_input_dim,
        hidden_units,
        target_output_dim,
    )
    optimizer = _optimizer(agent, intrinsic.learning_rate, intrinsic.optimizer)
    count_table_size = intrinsic.count_table_size if intrinsic.kind == "count" else 1
    count_key_dim = (
        _count_key_dim(input_dim, num_actions, intrinsic.action_conditioning)
        if intrinsic.kind == "count"
        else 1
    )
    return DqnIntrinsicState(
        target_params=target_params,
        prior_params=prior_params,
        predictor_params=predictor_params,
        opt_state=optimizer.init(predictor_params),
        reward_mean=jnp.asarray(0.0, dtype=jnp.float32),
        reward_var=jnp.asarray(1.0, dtype=jnp.float32),
        count_keys=jnp.zeros((count_table_size, count_key_dim), dtype=jnp.float32),
        counts=jnp.zeros((count_table_size,), dtype=jnp.float32),
        count_size=jnp.asarray(0, dtype=jnp.int32),
        count_overflow=jnp.asarray(False),
    )


def _sample_intrinsic_target(
    intrinsic: DqnIntrinsicConfig,
    key: jax.Array,
    target_dim: int,
) -> jax.Array:
    if intrinsic.kind != "cfn":
        return jnp.zeros((target_dim,), dtype=jnp.float32)
    targets = jax.random.bernoulli(key, p=0.5, shape=(target_dim,))
    return jnp.where(targets, 1.0, -1.0).astype(jnp.float32)


def _replay_intrinsic_target_dim(intrinsic: DqnIntrinsicConfig) -> int:
    if intrinsic.kind == "cfn":
        return intrinsic.output_dim
    return 1


def _replay_arrays(
    state: DqnReplayState,
    intrinsic: DqnIntrinsicConfig,
) -> dict[str, np.ndarray]:
    size = int(np.asarray(jax.device_get(state.size)))
    arrays = {
        "observations": np.asarray(jax.device_get(state.source_observations))[:size],
        "actions": np.asarray(jax.device_get(state.actions))[:size],
        "rewards": np.asarray(jax.device_get(state.rewards))[:size],
        "next_observations": np.asarray(jax.device_get(state.source_next_observations))[:size],
        "terminals": np.asarray(jax.device_get(state.terminals))[:size].astype(np.bool_),
    }
    if intrinsic.kind == "cfn":
        arrays["cfn_targets"] = np.asarray(jax.device_get(state.intrinsic_targets))[:size]
    return arrays


def _init_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_units: tuple[int, ...],
    output_dim: int,
) -> tuple[dict[str, jax.Array], ...]:
    dims = (input_dim, *hidden_units, output_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for layer_key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:], strict=True):
        scale = jnp.sqrt(2.0 / float(max(in_dim, 1)))
        params.append(
            {
                "w": jax.random.normal(layer_key, (in_dim, out_dim), dtype=jnp.float32) * scale,
                "b": jnp.zeros((out_dim,), dtype=jnp.float32),
            }
        )
    return tuple(params)


def _apply_mlp(
    params: tuple[dict[str, jax.Array], ...],
    observations: jax.Array,
    activation: str = "relu",
) -> jax.Array:
    x = jnp.asarray(observations, dtype=jnp.float32)
    for layer in params[:-1]:
        x = _activation(x @ layer["w"] + layer["b"], activation)
    output = params[-1]
    return x @ output["w"] + output["b"]


def _q_loss(
    agent: DqnAgentConfig,
    params,
    target_params,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
    known_mask: jax.Array,
    next_unknown_any: jax.Array,
) -> jax.Array:
    q_values = _apply_mlp(params, observations, agent.activation)
    selected_q = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
    if agent.double_q:
        next_online_q = _apply_mlp(params, next_observations, agent.activation)
        next_actions = jnp.argmax(jax.lax.stop_gradient(next_online_q), axis=1, keepdims=True)
        next_target_q = _apply_mlp(target_params, next_observations, agent.activation)
        next_q = jnp.take_along_axis(next_target_q, next_actions, axis=1).squeeze(-1)
    else:
        next_q = jnp.max(_apply_mlp(target_params, next_observations, agent.activation), axis=1)
    if agent.algorithm == "dqn_rmax":
        next_q = jnp.where(next_unknown_any, agent.rmax_v_max, next_q)
    target = rewards + agent.discount * next_q * (1.0 - terminals)
    td_error = selected_q - jax.lax.stop_gradient(target)
    losses = _td_loss(td_error, agent.loss_type, agent.huber_delta)
    if agent.algorithm == "dqn_rmax":
        weights = known_mask.astype(jnp.float32)
        return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.mean(losses)


def _rnd_prediction_error(
    target_params,
    predictor_params,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    intrinsic_input = _conditioned_input(
        observations,
        actions,
        intrinsic.action_conditioning,
        num_actions,
    )
    target_features = jax.lax.stop_gradient(
        _select_conditioned_features(
            _apply_mlp(target_params, intrinsic_input, intrinsic.activation),
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
    )
    predictor_features = _select_conditioned_features(
        _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
        actions,
        intrinsic.action_conditioning,
        intrinsic.output_dim,
        num_actions,
    )
    prediction_error = jnp.mean(jnp.square(predictor_features - target_features), axis=-1)
    return prediction_error, intrinsic_input, target_features


def _cfn_outputs(
    prior_params,
    predictor_params,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    intrinsic_input = _conditioned_input(
        observations,
        actions,
        intrinsic.action_conditioning,
        num_actions,
    )
    prior_features = _maybe_tanh(
        _apply_mlp(prior_params, intrinsic_input, intrinsic.activation),
        intrinsic.cfn_final_tanh,
    )
    prior_features = jax.lax.stop_gradient(
        _select_conditioned_features(
            prior_features,
            actions,
            intrinsic.action_conditioning,
            intrinsic.output_dim,
            num_actions,
        )
    )
    predictor_features = _maybe_tanh(
        _apply_mlp(predictor_params, intrinsic_input, intrinsic.activation),
        intrinsic.cfn_final_tanh,
    )
    predictor_features = _select_conditioned_features(
        predictor_features,
        actions,
        intrinsic.action_conditioning,
        intrinsic.output_dim,
        num_actions,
    )
    if intrinsic.cfn_use_random_prior:
        coin_flips = predictor_features + intrinsic.cfn_prior_scale * prior_features
    else:
        coin_flips = predictor_features
    raw_bonus = jnp.mean(jnp.square(coin_flips), axis=-1)
    raw_bonus = raw_bonus**intrinsic.cfn_bonus_exponent
    return raw_bonus, intrinsic_input, prior_features, predictor_features, coin_flips


def _conditioned_input(
    observations: jax.Array,
    actions: jax.Array,
    mode: ActionConditioning,
    num_actions: int,
) -> jax.Array:
    if mode == "input":
        return jnp.concatenate(
            (observations, jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)),
            axis=-1,
        )
    if mode == "pair":
        action_features = jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)
        pair_features = observations[..., :, None] * action_features[..., None, :]
        return pair_features.reshape(observations.shape[0], observations.shape[-1] * num_actions)
    return observations


def _select_conditioned_features(
    features: jax.Array,
    actions: jax.Array,
    mode: ActionConditioning,
    output_dim: int,
    num_actions: int,
) -> jax.Array:
    if mode != "output":
        return features
    features = features.reshape(features.shape[0], num_actions, output_dim)
    return jnp.take_along_axis(features, actions[:, None, None], axis=1).squeeze(axis=1)


def _normalize_intrinsic_reward(
    intrinsic: DqnIntrinsicConfig,
    raw_bonus: jax.Array,
    reward_mean: jax.Array,
    reward_var: jax.Array,
) -> jax.Array:
    if intrinsic.intrinsic_reward_center:
        raw_bonus = raw_bonus - reward_mean
    reward_scale = jnp.sqrt(jnp.maximum(reward_var, intrinsic.intrinsic_reward_epsilon))
    normalized = raw_bonus / reward_scale
    if intrinsic.intrinsic_reward_clip is not None:
        normalized = jnp.clip(normalized, 0.0, intrinsic.intrinsic_reward_clip)
    # TODO: expose option of keeping negative rewards or shifting them to be non-negative
    else:
        normalized = normalized - jnp.minimum(jnp.min(normalized), 0.0)
    return normalized


def _update_intrinsic_stats(
    intrinsic: DqnIntrinsicConfig,
    old_mean: jax.Array,
    old_var: jax.Array,
    raw_bonus: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_mean = jnp.mean(raw_bonus)
    batch_second_moment = jnp.mean(jnp.square(raw_bonus))
    old_second_moment = old_var + jnp.square(old_mean)
    decay = intrinsic.intrinsic_stats_decay
    new_mean = decay * old_mean + (1.0 - decay) * batch_mean
    new_second_moment = decay * old_second_moment + (1.0 - decay) * batch_second_moment
    new_var = jnp.maximum(
        new_second_moment - jnp.square(new_mean),
        intrinsic.intrinsic_reward_epsilon,
    )
    return new_mean, new_var


def _epsilon(
    step: jax.Array,
    start: float,
    end: float,
    decay_steps: int,
) -> jax.Array:
    fraction = jnp.minimum(1.0, step.astype(jnp.float32) / float(decay_steps))
    return start + fraction * (end - start)


def _epsilon_greedy_action(
    params,
    observation: jax.Array,
    key: jax.Array,
    epsilon: jax.Array,
    num_actions: int,
    activation: str,
) -> jax.Array:
    greedy_key, random_key, choice_key = jax.random.split(key, 3)
    del greedy_key
    q_values = _apply_mlp(params, observation[None, :], activation).squeeze(0)
    greedy_action = jnp.argmax(q_values).astype(jnp.int32)
    random_action = jax.random.randint(random_key, (), 0, num_actions, dtype=jnp.int32)
    explore = jax.random.uniform(choice_key) < epsilon
    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def _select_action(
    agent: DqnAgentConfig,
    params,
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    observation: jax.Array,
    key: jax.Array,
    num_actions: int,
    global_step: jax.Array,
    *,
    training: bool,
) -> jax.Array:
    if agent.algorithm == "dqn_rmax":
        return _rmax_action(
            agent,
            params,
            intrinsic_state,
            intrinsic,
            observation,
            key,
            num_actions,
        )
    epsilon = (
        _epsilon(
            global_step,
            agent.epsilon_start,
            agent.epsilon_end,
            agent.epsilon_decay_steps,
        )
        if training
        else jnp.asarray(agent.eval_epsilon, dtype=jnp.float32)
    )
    return _epsilon_greedy_action(
        params,
        observation,
        key,
        epsilon,
        num_actions,
        agent.activation,
    )


def _rmax_action(
    agent: DqnAgentConfig,
    params,
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    observation: jax.Array,
    key: jax.Array,
    num_actions: int,
) -> jax.Array:
    q_values = _apply_mlp(params, observation[None, :], agent.activation).squeeze(0)
    bonuses = _intrinsic_bonus_for_all_actions(
        intrinsic_state,
        observation[None, :],
        intrinsic,
        num_actions,
    ).squeeze(0)
    optimistic_values = jnp.where(
        bonuses > agent.rmax_bonus_threshold,
        agent.rmax_v_max,
        q_values,
    )
    max_value = jnp.max(optimistic_values)
    ties = optimistic_values == max_value
    tie_count = jnp.sum(ties.astype(jnp.int32))
    selected_tie = jax.random.randint(key, (), 0, tie_count, dtype=jnp.int32)
    tie_offsets = jnp.cumsum(ties.astype(jnp.int32)) - 1
    return jnp.argmax(jnp.logical_and(ties, tie_offsets == selected_tie)).astype(jnp.int32)


def _rmax_batch_masks(
    intrinsic_state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
    agent: DqnAgentConfig,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    num_actions: int,
) -> tuple[jax.Array, jax.Array]:
    if agent.algorithm != "dqn_rmax":
        return (
            jnp.ones_like(actions, dtype=jnp.bool_),
            jnp.zeros_like(actions, dtype=jnp.bool_),
        )
    current_bonus = _intrinsic_bonus(
        intrinsic_state,
        observations,
        actions,
        intrinsic,
        num_actions,
    )
    next_bonuses = _intrinsic_bonus_for_all_actions(
        intrinsic_state,
        next_observations,
        intrinsic,
        num_actions,
    )
    return (
        current_bonus <= agent.rmax_bonus_threshold,
        jnp.any(next_bonuses > agent.rmax_bonus_threshold, axis=1),
    )


def _intrinsic_bonus_for_all_actions(
    state: DqnIntrinsicState,
    observations: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    batch_size = observations.shape[0]
    actions = jnp.broadcast_to(
        jnp.arange(num_actions, dtype=jnp.int32),
        (batch_size, num_actions),
    ).reshape(-1)
    repeated_observations = jnp.repeat(observations, num_actions, axis=0)
    bonuses = _intrinsic_bonus(
        state,
        repeated_observations,
        actions,
        intrinsic,
        num_actions,
    )
    return bonuses.reshape(batch_size, num_actions)


def _intrinsic_bonus(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    if intrinsic.kind == "none":
        return jnp.zeros_like(actions, dtype=jnp.float32)
    if intrinsic.kind == "rnd":
        raw_bonus, _intrinsic_input, _target_features = _rnd_prediction_error(
            state.target_params,
            state.predictor_params,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    elif intrinsic.kind == "cfn":
        raw_bonus, _intrinsic_input, _prior_features, _predictor_features, _coin_flips = _cfn_outputs(
            state.prior_params,
            state.predictor_params,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    else:
        # TODO: Implement flag to control normalization. count-based rewards I'm not normalizing rn.
        return _count_raw_bonus(
            state,
            observations,
            actions,
            intrinsic,
            num_actions,
        )
    return _normalize_intrinsic_reward(
        intrinsic,
        raw_bonus,
        state.reward_mean,
        state.reward_var,
    )


def _observe_intrinsic_transition(
    state: DqnIntrinsicState,
    observation: jax.Array,
    action: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> DqnIntrinsicState:
    if intrinsic.kind != "count":
        return state
    key = _count_keys(
        observation[None, :],
        action.reshape((1,)),
        intrinsic,
        num_actions,
    )[0]
    index, found = _count_lookup_one(state, key)
    has_capacity = state.count_size < state.counts.shape[0]
    should_insert = jnp.logical_and(~found, has_capacity)
    should_record = jnp.logical_or(found, has_capacity)
    write_index = jnp.where(found, index, state.count_size)
    safe_index = jnp.minimum(write_index, state.counts.shape[0] - 1).astype(jnp.int32)
    existing_key = state.count_keys[safe_index]
    count_keys = state.count_keys.at[safe_index].set(
        jnp.where(should_insert, key, existing_key)
    )
    counts = state.counts.at[safe_index].add(should_record.astype(jnp.float32))
    return state._replace(
        count_keys=count_keys,
        counts=counts,
        count_size=state.count_size + should_insert.astype(jnp.int32),
        count_overflow=jnp.logical_or(
            state.count_overflow,
            jnp.logical_and(~found, ~has_capacity),
        ),
    )


def _count_raw_bonus(
    state: DqnIntrinsicState,
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    keys = _count_keys(observations, actions, intrinsic, num_actions)
    indices, found = _count_lookup(state, keys)
    counts = jnp.where(found, state.counts[indices], 0.0)
    effective_counts = jnp.maximum(counts, intrinsic.count_min_count)
    return 1.0 / (effective_counts**intrinsic.count_bonus_exponent)


def _count_lookup(
    state: DqnIntrinsicState,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    valid = jnp.arange(state.counts.shape[0], dtype=jnp.int32) < state.count_size
    matches = jnp.all(state.count_keys[None, :, :] == keys[:, None, :], axis=-1)
    matches = jnp.logical_and(matches, valid[None, :])
    found = jnp.any(matches, axis=1)
    indices = jnp.argmax(matches.astype(jnp.int32), axis=1).astype(jnp.int32)
    return indices, found


def _count_lookup_one(
    state: DqnIntrinsicState,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    indices, found = _count_lookup(state, key[None, :])
    return indices[0], found[0]


def _count_keys(
    observations: jax.Array,
    actions: jax.Array,
    intrinsic: DqnIntrinsicConfig,
    num_actions: int,
) -> jax.Array:
    if intrinsic.action_conditioning == "none":
        return observations.astype(jnp.float32)
    if intrinsic.action_conditioning == "pair":
        return _conditioned_input(
            observations,
            actions,
            "pair",
            num_actions,
        ).astype(jnp.float32)
    return jnp.concatenate(
        (observations, jax.nn.one_hot(actions, num_actions, dtype=jnp.float32)),
        axis=-1,
    ).astype(jnp.float32)


def _count_key_dim(
    input_dim: int,
    num_actions: int,
    mode: ActionConditioning,
) -> int:
    if mode == "none":
        return input_dim
    if mode == "pair":
        return input_dim * num_actions
    return input_dim + num_actions


def _td_loss(td_error: jax.Array, loss_type: str, huber_delta: float) -> jax.Array:
    if loss_type == "mse":
        return jnp.square(td_error)
    abs_error = jnp.abs(td_error)
    quadratic = jnp.minimum(abs_error, huber_delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + huber_delta * linear


def _activation(x: jax.Array, name: str) -> jax.Array:
    if name == "tanh":
        return jnp.tanh(x)
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "elu":
        return jax.nn.elu(x)
    if name == "linear":
        return x
    return jax.nn.relu(x)


def _optimizer(
    agent: DqnAgentConfig,
    learning_rate: float,
    optimizer_name: str,
) -> optax.GradientTransformation:
    if optimizer_name == "sgd":
        base = optax.sgd(learning_rate, momentum=agent.optimizer_momentum)
    elif optimizer_name == "rmsprop":
        base = optax.rmsprop(
            learning_rate,
            decay=agent.optimizer_decay,
            eps=agent.optimizer_epsilon,
            momentum=agent.optimizer_momentum,
            centered=agent.optimizer_centered,
        )
    else:
        base = optax.adamw(
            learning_rate,
            b1=agent.optimizer_beta1,
            b2=agent.optimizer_beta2,
            eps=agent.optimizer_epsilon,
            weight_decay=agent.optimizer_weight_decay,
        )
    if agent.max_grad_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(agent.max_grad_norm), base)
    return base


def _input_dim_from_space(space: Any) -> int:
    shape = tuple(space.shape)
    if shape in {(), (1,)} and hasattr(space, "n"):
        return int(space.n)
    return int(np.prod(np.asarray(shape)))


def _integer_observation_scale(dtype: np.dtype) -> float | None:
    if not np.issubdtype(dtype, np.integer):
        return None
    info = np.iinfo(dtype)
    return float(info.max) if info.max > 0 else None


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


def _count_table_status(
    state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
) -> tuple[int | None, bool | None]:
    if intrinsic.kind != "count":
        return None, None
    return (
        int(np.asarray(jax.device_get(state.count_size))),
        bool(np.asarray(jax.device_get(state.count_overflow))),
    )


def _handle_count_table_overflow(
    intrinsic: DqnIntrinsicConfig,
    overflow: bool | None,
) -> None:
    if intrinsic.kind != "count" or not overflow:
        return
    message = (
        "Exact count table exceeded count_table_size="
        f"{intrinsic.count_table_size}; additional novel count keys were not inserted. "
        "Increase count_table_size or set count_table_overflow='error' to fail runs."
    )
    if intrinsic.count_table_overflow == "error":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _conditioned_input_dim(input_dim: int, num_actions: int, mode: ActionConditioning) -> int:
    if mode == "input":
        return input_dim + num_actions
    if mode == "pair":
        return input_dim * num_actions
    return input_dim


def _conditioned_output_dim(num_actions: int, output_dim: int, mode: ActionConditioning) -> int:
    if mode == "output":
        return num_actions * output_dim
    return output_dim


def _clone_params(params):
    return jax.tree_util.tree_map(lambda item: jnp.array(item, copy=True), params)


def _maybe_hard_update(source, target, should_update: jax.Array):
    return jax.tree_util.tree_map(
        lambda source_leaf, target_leaf: jnp.where(should_update, source_leaf, target_leaf),
        source,
        target,
    )


def _maybe_tanh(features: jax.Array, enabled: bool) -> jax.Array:
    if enabled:
        return jnp.tanh(features)
    return features


def _aux_params(
    state: DqnIntrinsicState,
    intrinsic: DqnIntrinsicConfig,
) -> dict[str, tuple[dict[str, jax.Array], ...]]:
    if intrinsic.kind == "rnd":
        return {
            "rnd_target": state.target_params,
            "rnd_predictor": state.predictor_params,
        }
    if intrinsic.kind == "cfn":
        return {
            "cfn_prior": state.prior_params,
            "cfn_predictor": state.predictor_params,
        }
    return {}


def _coerce_navix_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_name": settings["env_name"],
        "size": int(settings["size"]),
        "layout": settings["layout"],
        "observation_mode": settings["observation_mode"],
        "action_set": settings["action_set"],
        "max_steps": settings["max_steps"],
    }


def _timestep_done(timestep: Any) -> jax.Array:
    if hasattr(timestep, "is_done"):
        return timestep.is_done()
    return jnp.logical_or(timestep.is_termination(), timestep.is_truncation())


def _resolve_replay_save_path(path: str, run_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    if candidate.is_absolute():
        return candidate
    return (run_dir / candidate).resolve()
