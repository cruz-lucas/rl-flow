from __future__ import annotations

from typing import Any

from rlflow_builtin.tabular.types import RunnerConfig


def runner_config(config: dict[str, Any]) -> RunnerConfig:
    checkpoint_freq = config["checkpoint_freq"]
    train_steps = config.get("train_steps")
    return RunnerConfig(
        seed=int(config["seed"]),
        train_episodes=int(config["train_episodes"]),
        train_steps=None if train_steps is None else int(train_steps),
        max_episode_steps=int(config["max_episode_steps"]),
        eval_episodes=int(config["eval_episodes"]),
        checkpoint_freq=None if checkpoint_freq is None else int(checkpoint_freq),
        checkpoint_dir=str(config["checkpoint_dir"]),
        save_final_checkpoint=bool(config["save_final_checkpoint"]),
    )
