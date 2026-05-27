"""Navix environment wrappers and observation encoders."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
from flax import struct
from jax import Array
from navix import actions, observations, rewards, terminations
from navix.actions import _can_walk_there
from navix.components import EMPTY_POCKET_ID
from navix.entities import Door, Goal, Key, Player, Wall
from navix.environments import DoorKey
from navix.environments.environment import Timestep
from navix.environments.empty import Room as EmptyRoom
from navix.grid import room, translate
from navix.rendering.cache import RenderingCache
from navix.rendering.registry import PALETTE
from navix.spaces import Discrete, Space
from navix.states import State

try:
    import gin
except ImportError:  # pragma: no cover - Gin is optional for generated configs.
    gin = None


NavixEnvName = Literal["empty_room", "doorkey"]
NavixLayout = Literal["fixed", "random", "layout1", "layout2", "layout3"]
NavixObservationMode = Literal["tabular", "one_hot", "state_features", "symbolic", "rgb"]
NavixActionSet = Literal["default", "cardinal"]
NavixSymbolicDistractor = Literal[
    "none",
    "corner_wall_color",
    "shared_wall_color",
    "independent_wall_color",
]

SUPPORTED_SIZES = (5, 6, 8, 16)
FIXED_DOORKEY_LAYOUTS: dict[int, dict[str, dict[str, int]]] = {
    5: {
        "layout1": {"door_row": 1, "door_col": 2, "key_row": 2, "key_col": 1, "goal_row": 1, "goal_col": 3},
        "layout2": {"door_row": 3, "door_col": 2, "key_row": 2, "key_col": 1, "goal_row": 3, "goal_col": 3},
        "layout3": {"door_row": 3, "door_col": 2, "key_row": 2, "key_col": 1, "goal_row": 1, "goal_col": 3},
    },
    16: {
        "layout1": {"door_row": 1, "door_col": 13, "key_row": 2, "key_col": 1, "goal_row": 1, "goal_col": 14},
        "layout2": {"door_row": 14, "door_col": 13, "key_row": 2, "key_col": 1, "goal_row": 14, "goal_col": 14},
        "layout3": {"door_row": 14, "door_col": 13, "key_row": 2, "key_col": 1, "goal_row": 1, "goal_col": 14},
    },
}


@dataclass(frozen=True)
class NavixSpec:
    env_name: NavixEnvName
    height: int
    width: int
    layout: NavixLayout
    observation_mode: NavixObservationMode
    action_set: NavixActionSet
    symbolic_distractor: NavixSymbolicDistractor = "none"

    @property
    def inner_states(self) -> int:
        return (self.height - 2) * (self.width - 2)

    @property
    def fixed_layout(self) -> bool:
        return self.layout in {"fixed", "layout1", "layout2", "layout3"}

    @property
    def uses_direction(self) -> bool:
        return self.env_name == "doorkey" or self.action_set != "cardinal"


def _move_absolute(state: State, direction: int) -> State:
    player = state.get_player(idx=0)
    absolute_direction = jnp.asarray(direction, dtype=jnp.int32)
    target_position = translate(player.position, absolute_direction)
    can_move, events = _can_walk_there(state, target_position)
    next_position = jnp.where(can_move, target_position, player.position)
    player = player.replace(position=next_position)
    return state.set_player(player).replace(events=events)


def _move_right(state: State) -> State:
    return _move_absolute(state, 0)


def _move_down(state: State) -> State:
    return _move_absolute(state, 1)


def _move_left(state: State) -> State:
    return _move_absolute(state, 2)


def _move_up(state: State) -> State:
    return _move_absolute(state, 3)


CARDINAL_ACTION_SET = (_move_up, _move_down, _move_left, _move_right)


class FixedLayoutDoorKey(DoorKey):
    door_row: int = struct.field(pytree_node=False, default=1)
    door_col: int = struct.field(pytree_node=False, default=2)
    key_row: int = struct.field(pytree_node=False, default=2)
    key_col: int = struct.field(pytree_node=False, default=1)
    goal_row: int = struct.field(pytree_node=False, default=1)
    goal_col: int = struct.field(pytree_node=False, default=3)

    def _reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        if self.height <= 3:
            raise ValueError(f"Room height must be greater than 3, got {self.height}")
        if self.width <= 4:
            raise ValueError(f"Room width must be greater than 4, got {self.width}")

        key, _unused = jax.random.split(key)
        grid = room(height=self.height, width=self.width)

        door_pos = jnp.asarray((self.door_row, self.door_col), dtype=jnp.int32)
        doors = Door.create(
            position=door_pos,
            requires=jnp.asarray(3),
            open=jnp.asarray(False),
            colour=PALETTE.YELLOW,
        )

        wall_rows = jnp.arange(1, self.height - 1, dtype=jnp.int32)
        wall_cols = jnp.full((self.height - 2,), self.door_col, dtype=jnp.int32)
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        wall_pos = jnp.delete(wall_pos, self.door_row - 1, axis=0, assume_unique_indices=True)
        walls = Wall.create(position=wall_pos)

        player = Player.create(
            position=jnp.asarray([1, 1], dtype=jnp.int32),
            direction=jnp.asarray(0, dtype=jnp.int32),
            pocket=EMPTY_POCKET_ID,
        )
        goals = Goal.create(
            position=jnp.asarray([self.goal_row, self.goal_col], dtype=jnp.int32),
            probability=jnp.asarray(1.0),
        )
        keys = Key.create(
            position=jnp.asarray([self.key_row, self.key_col], dtype=jnp.int32),
            id=jnp.asarray(3),
            colour=PALETTE.YELLOW,
        )

        grid = grid.at[tuple(door_pos)].set(0)
        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities={
                "player": player[None],
                "key": keys[None],
                "door": doors[None],
                "goal": goals[None],
                "wall": walls,
            },
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def create_navix_environment(
    env_name: NavixEnvName = "empty_room",
    size: int = 5,
    layout: NavixLayout = "fixed",
    observation_mode: NavixObservationMode = "symbolic",
    action_set: NavixActionSet = "default",
    max_steps: int | None = None,
    symbolic_distractor: NavixSymbolicDistractor = "none",
) -> nx.Environment:
    spec = _validate_spec(
        env_name,
        size,
        layout,
        observation_mode,
        action_set,
        symbolic_distractor,
    )
    observation_fn, observation_space = _observation_config(spec)
    resolved_action_set = _action_set(spec)
    max_steps = max_steps or 4 * size * size

    kwargs: dict[str, Any] = {
        "max_steps": max_steps,
        "observation_fn": observation_fn,
        "action_set": resolved_action_set,
        "reward_fn": rewards.on_goal_reached,
        "termination_fn": terminations.on_goal_reached,
    }
    if observation_space is not None:
        kwargs["observation_space"] = observation_space

    env: nx.Environment
    if spec.env_name == "empty_room":
        env = EmptyRoom.create(
            height=size,
            width=size,
            random_start=spec.layout == "random",
            **kwargs,
        )
    elif spec.fixed_layout:
        layout_name = "layout1" if spec.layout == "fixed" else spec.layout
        layout_kwargs = FIXED_DOORKEY_LAYOUTS[size][layout_name]
        env = FixedLayoutDoorKey.create(
            height=size,
            width=size,
            random_start=False,
            **layout_kwargs,
            **kwargs,
        )
    else:
        env = nx.make(f"Navix-DoorKey-Random-{size}x{size}-v0", **kwargs)

    if spec.symbolic_distractor != "none":
        return FreshKeyObservationEnv(env)
    return env


if gin is not None:
    create_navix_environment = gin.configurable(create_navix_environment)


class NavixWrapper:
    """Thin wrapper exposing Navix reset/step results in rl-flow's JAX env shape."""

    def __init__(
        self,
        env_name: NavixEnvName = "empty_room",
        size: int = 5,
        layout: NavixLayout = "fixed",
        observation_mode: NavixObservationMode = "symbolic",
        action_set: NavixActionSet = "default",
        max_steps: int | None = None,
        symbolic_distractor: NavixSymbolicDistractor = "none",
    ) -> None:
        self.env = create_navix_environment(
            env_name=env_name,
            size=size,
            layout=layout,
            observation_mode=observation_mode,
            action_set=action_set,
            max_steps=max_steps,
            symbolic_distractor=symbolic_distractor,
        )
        self.observation_shape = tuple(self.env.observation_space.shape)
        self.observation_dtype = np.dtype(self.env.observation_space.dtype)
        self.num_observation_states = (
            int(self.env.observation_space.n)
            if self.observation_shape in {(), (1,)}
            else int(np.prod(np.asarray(self.observation_shape)))
        )

    def reset(self, key: Array) -> tuple[Timestep, Array]:
        timestep = self.env.reset(key)
        return timestep, timestep.observation

    def step(self, timestep: Timestep, action: Array) -> tuple[Timestep, Array, Array, Array, Array, dict[str, Any]]:
        timestep = self.env.step(timestep, jnp.asarray(action).reshape(()))
        return (
            timestep,
            timestep.observation,
            timestep.reward,
            timestep.is_termination(),
            timestep.is_truncation(),
            timestep.info,
        )


def _validate_spec(
    env_name: str,
    size: int,
    layout: str,
    observation_mode: str,
    action_set: str,
    symbolic_distractor: str,
) -> NavixSpec:
    if env_name not in {"empty_room", "doorkey"}:
        raise ValueError("Navix env_name must be 'empty_room' or 'doorkey'")
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"Navix size must be one of {SUPPORTED_SIZES}")
    if layout not in {"fixed", "random", "layout1", "layout2", "layout3"}:
        raise ValueError("Navix layout must be fixed, random, layout1, layout2, or layout3")
    if observation_mode not in {"tabular", "one_hot", "state_features", "symbolic", "rgb"}:
        raise ValueError("Unsupported Navix observation_mode")
    if action_set not in {"default", "cardinal"}:
        raise ValueError("Navix action_set must be 'default' or 'cardinal'")
    if symbolic_distractor not in {
        "none",
        "corner_wall_color",
        "shared_wall_color",
        "independent_wall_color",
    }:
        raise ValueError("Unsupported Navix symbolic_distractor")
    if symbolic_distractor != "none" and env_name != "empty_room":
        raise ValueError("Symbolic distractors are only supported for empty_room")
    if symbolic_distractor != "none" and observation_mode != "symbolic":
        raise ValueError("Symbolic distractors require observation_mode='symbolic'")
    if env_name == "doorkey" and action_set == "cardinal":
        raise ValueError("The cardinal action set is only supported for empty_room")
    if env_name == "empty_room" and layout in {"layout1", "layout2", "layout3"}:
        raise ValueError("Empty-room Navix layouts are fixed or random")
    if env_name == "doorkey" and layout != "random":
        layout_name = "layout1" if layout == "fixed" else layout
        if size not in FIXED_DOORKEY_LAYOUTS or layout_name not in FIXED_DOORKEY_LAYOUTS[size]:
            raise ValueError("Fixed DoorKey layouts are currently available for 5x5 and 16x16 only")
    return NavixSpec(
        env_name=env_name,  # type: ignore[arg-type]
        height=size,
        width=size,
        layout=layout,  # type: ignore[arg-type]
        observation_mode=observation_mode,  # type: ignore[arg-type]
        action_set=action_set,  # type: ignore[arg-type]
        symbolic_distractor=symbolic_distractor,  # type: ignore[arg-type]
    )


def _action_set(spec: NavixSpec):
    if spec.action_set == "cardinal":
        return CARDINAL_ACTION_SET
    return actions.DEFAULT_ACTION_SET


def _observation_config(spec: NavixSpec) -> tuple[Callable[[State], Array], Space | None]:
    if spec.observation_mode == "symbolic":
        if spec.symbolic_distractor != "none":
            return (
                partial(symbolic_distractor_observation, mode=spec.symbolic_distractor),
                Discrete.create(
                    256,
                    shape=(spec.height, spec.width, 3),
                    dtype=jnp.uint8,
                ),
            )
        return observations.symbolic, None
    if spec.observation_mode == "rgb":
        return observations.rgb, None
    if spec.observation_mode == "tabular":
        return partial(tabular_observation, spec=spec), Discrete.create(
            _state_space_size(spec),
            dtype=jnp.int32,
        )
    if spec.observation_mode == "one_hot":
        return partial(one_hot_observation, spec=spec), Discrete.create(
            2,
            shape=(_state_space_size(spec),),
            dtype=jnp.float32,
        )
    return partial(state_features_observation, spec=spec), Discrete.create(
        2,
        shape=(_feature_size(spec),),
        dtype=jnp.float32,
    )


class FreshKeyObservationEnv:
    """Recomputes observations with a freshly split state key on every timestep."""

    def __init__(self, env: nx.Environment) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        return self._refresh_observation(self.env.reset(key, cache))

    def step(self, timestep: Timestep, action: Array) -> Timestep:
        return self._refresh_observation(self.env.step(timestep, action))

    def _refresh_observation(self, timestep: Timestep) -> Timestep:
        observation_key, next_key = jax.random.split(timestep.state.key)
        observation_state = timestep.state.replace(key=observation_key)
        observation = self.env.observation_fn(observation_state)
        return timestep.replace(
            state=observation_state.replace(key=next_key),
            observation=observation,
        )


def symbolic_distractor_observation(state: State, *, mode: NavixSymbolicDistractor) -> Array:
    obs = observations.symbolic(state)
    wall_mask = state.grid == -1
    if mode == "corner_wall_color":
        value = jax.random.randint(state.key, (), 0, 256, dtype=jnp.uint8)
        return obs.at[0, -1, 1].set(value)
    if mode == "shared_wall_color":
        value = jax.random.randint(state.key, (), 0, 256, dtype=jnp.uint8)
        return obs.at[..., 1].set(jnp.where(wall_mask, value, obs[..., 1]))
    values = jax.random.randint(state.key, state.grid.shape, 0, 256, dtype=jnp.uint8)
    return obs.at[..., 1].set(jnp.where(wall_mask, values, obs[..., 1]))


def tabular_observation(state: State, *, spec: NavixSpec) -> Array:
    if spec.env_name == "empty_room":
        player = state.get_player(idx=0)
        index = _position_index(player.position, spec)
        if spec.uses_direction:
            index = index * 4 + player.direction.astype(jnp.int32)
        return index.astype(jnp.int32)
    return _doorkey_index(state, spec).astype(jnp.int32)


def one_hot_observation(state: State, *, spec: NavixSpec) -> Array:
    return jax.nn.one_hot(tabular_observation(state, spec=spec), _state_space_size(spec), dtype=jnp.float32)


def state_features_observation(state: State, *, spec: NavixSpec) -> Array:
    if spec.env_name == "empty_room":
        player = state.get_player(idx=0)
        features = [jax.nn.one_hot(_position_index(player.position, spec), spec.inner_states, dtype=jnp.float32)]
        if spec.uses_direction:
            features.append(jax.nn.one_hot(player.direction.astype(jnp.int32), 4, dtype=jnp.float32))
        return jnp.concatenate(features)

    player = state.get_player(idx=0)
    key_pos = state.get_keys().position[0]
    door = state.get_doors()
    door_pos = door.position[0]
    features = [
        jax.nn.one_hot(_position_index(player.position, spec), spec.inner_states, dtype=jnp.float32),
        jax.nn.one_hot(_key_index(key_pos, spec), spec.inner_states + 1, dtype=jnp.float32),
        jax.nn.one_hot(door.open[0].astype(jnp.int32), 2, dtype=jnp.float32),
        jax.nn.one_hot(player.direction.astype(jnp.int32), 4, dtype=jnp.float32),
    ]
    if not spec.fixed_layout:
        features.append(jax.nn.one_hot(_position_index(door_pos, spec), spec.inner_states, dtype=jnp.float32))
    return jnp.concatenate(features)


def _doorkey_index(state: State, spec: NavixSpec) -> Array:
    player = state.get_player(idx=0)
    key_pos = state.get_keys().position[0]
    door = state.get_doors()
    index = _key_index(key_pos, spec)
    if not spec.fixed_layout:
        index = index + _position_index(door.position[0], spec) * (spec.inner_states + 1)
    index = index * spec.inner_states + _position_index(player.position, spec)
    index = index * 2 + door.open[0].astype(jnp.int32)
    return index * 4 + player.direction.astype(jnp.int32)


def _position_index(position: Array, spec: NavixSpec) -> Array:
    row = jnp.clip(position[0] - 1, 0, spec.height - 3)
    col = jnp.clip(position[1] - 1, 0, spec.width - 3)
    return (row * (spec.width - 2) + col).astype(jnp.int32)


def _key_index(position: Array, spec: NavixSpec) -> Array:
    picked = jnp.any(position < 0)
    return jnp.where(picked, 0, _position_index(position, spec) + 1).astype(jnp.int32)


def _state_space_size(spec: NavixSpec) -> int:
    if spec.env_name == "empty_room":
        return spec.inner_states * (4 if spec.uses_direction else 1)
    size = (spec.inner_states + 1) * spec.inner_states * 2 * 4
    if not spec.fixed_layout:
        size *= spec.inner_states
    return size


def _feature_size(spec: NavixSpec) -> int:
    if spec.env_name == "empty_room":
        return spec.inner_states + (4 if spec.uses_direction else 0)
    size = spec.inner_states + (spec.inner_states + 1) + 2 + 4
    if not spec.fixed_layout:
        size += spec.inner_states
    return size
