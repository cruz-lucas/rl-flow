"""Vectorised Atari environments for the delegate loop.

Two backends behind one tiny batched interface:

* ``envpool`` — the fast, vectorised C++ ALE (Linux/cluster only; imported
  lazily so the module stays importable on macOS/CI where envpool has no wheel).
* ``synthetic`` — a dependency-free uint8 image env with the same API, used by
  unit tests and local development where envpool is unavailable.

Interface (numpy, channels-first stacked frames ``(num_envs, C, H, W)``):
    reset() -> obs
    step(actions) -> (obs, rewards, terminals, truncations)
Both backends auto-reset finished envs; on a done step the returned obs is the
first frame of the next episode (terminal masks its bootstrap value anyway).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AtariEnvConfig:
    task_id: str
    backend: str
    num_envs: int
    frame_skip: int
    frame_stack: int
    img_height: int
    img_width: int
    gray_scale: bool
    repeat_action_probability: float
    episodic_life: bool
    noop_max: int
    max_episode_steps: int | None
    reward_clip: bool
    full_action_space: bool


def atari_env_config(resolved: dict[str, Any]) -> AtariEnvConfig:
    return AtariEnvConfig(
        task_id=str(resolved.get("task_id", "MontezumaRevenge-v5")),
        backend=str(resolved.get("backend", "envpool")),
        num_envs=int(resolved.get("num_envs", 8)),
        frame_skip=int(resolved.get("frame_skip", 4)),
        frame_stack=int(resolved.get("frame_stack", 4)),
        img_height=int(resolved.get("img_height", 84)),
        img_width=int(resolved.get("img_width", 84)),
        gray_scale=bool(resolved.get("gray_scale", True)),
        repeat_action_probability=float(resolved.get("repeat_action_probability", 0.25)),
        episodic_life=bool(resolved.get("episodic_life", False)),
        noop_max=int(resolved.get("noop_max", 30)),
        max_episode_steps=resolved.get("max_episode_steps"),
        reward_clip=bool(resolved.get("reward_clip", True)),
        full_action_space=bool(resolved.get("full_action_space", False)),
    )


class EnvpoolAtari:
    """Adapter over an envpool gymnasium vector env."""

    def __init__(self, config: AtariEnvConfig, seed: int) -> None:
        import envpool  # lazy: Linux-only dependency

        self._envs = envpool.make(
            config.task_id,
            env_type="gymnasium",
            num_envs=config.num_envs,
            seed=int(seed),
            stack_num=config.frame_stack,
            frame_skip=config.frame_skip,
            gray_scale=config.gray_scale,
            img_height=config.img_height,
            img_width=config.img_width,
            repeat_action_probability=config.repeat_action_probability,
            episodic_life=config.episodic_life,
            noop_max=config.noop_max,
            max_episode_steps=int(config.max_episode_steps or 108_000),
            # We clip rewards in-agent for the TD target and log the raw score,
            # so envpool must return the unclipped reward.
            reward_clip=False,
            full_action_space=config.full_action_space,
        )
        self.num_envs = int(config.num_envs)
        self.num_actions = int(self._envs.action_space.n)
        self.observation_shape = tuple(int(dim) for dim in self._envs.observation_space.shape)

    def reset(self) -> np.ndarray:
        obs, _info = self._envs.reset()
        return np.asarray(obs, dtype=np.uint8)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = self._envs.step(np.asarray(actions, dtype=np.int32))
        if len(result) == 5:
            obs, rewards, terminated, truncated, _info = result
        else:  # older gym 4-tuple
            obs, rewards, terminated, _info = result
            truncated = np.zeros_like(terminated)
        return (
            np.asarray(obs, dtype=np.uint8),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
        )


class SyntheticAtari:
    """Dependency-free stand-in with the same batched API (for tests/CI)."""

    def __init__(self, config: AtariEnvConfig, seed: int) -> None:
        self.num_envs = int(config.num_envs)
        self.num_actions = 6
        self.observation_shape = (
            int(config.frame_stack),
            int(config.img_height),
            int(config.img_width),
        )
        self._episode_len = int(config.max_episode_steps or 20)
        self._rng = np.random.default_rng(seed)
        self._t = np.zeros(self.num_envs, dtype=np.int64)

    def _observe(self) -> np.ndarray:
        return self._rng.integers(
            0, 256, size=(self.num_envs, *self.observation_shape), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        self._t[:] = 0
        return self._observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del actions
        self._t += 1
        obs = self._observe()
        rewards = (self._rng.random(self.num_envs) < 0.05).astype(np.float32)
        terminated = self._t >= self._episode_len
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self._t = np.where(terminated, 0, self._t)
        return obs, rewards, terminated, truncated


def make_atari_env(config: AtariEnvConfig, seed: int) -> EnvpoolAtari | SyntheticAtari:
    if config.backend == "synthetic":
        return SyntheticAtari(config, seed)
    if config.backend == "envpool":
        return EnvpoolAtari(config, seed)
    raise ValueError(f"Unsupported atari env backend: {config.backend!r}")
