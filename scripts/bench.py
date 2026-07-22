"""Lightweight performance benchmark for the JAX training loops.

Measures in-process throughput (JIT compile time + environment steps/second) for a
fixed tabular run so refactors that regress the hot loop are caught. Writes a JSON
report; compare successive runs on the same machine rather than across hardware.

    uv run python scripts/bench.py            # print report
    uv run python scripts/bench.py --out bench.json
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from rlflow_builtin.tabular.training import run_tabular_training
from rlflow_builtin.tabular.types import (
    AgentConfig,
    EnvironmentConfig,
    PolicyConfig,
    RunnerConfig,
)

_RIVERSWIM = EnvironmentConfig(
    name="riverswim",
    num_states=6,
    num_actions=2,
    start_state=0,
    p_left=0.1,
    p_stay=0.6,
    p_right=0.3,
    easy_reward=0.005,
    hard_reward=1.0,
    step_reward=0.0,
)


def _run(runner: RunnerConfig) -> Any:
    agent = AgentConfig(algorithm="q_learning", discount=0.99, learning_rate=0.1, initial_q=0.0)
    policy = PolicyConfig(name="epsilon_greedy", epsilon=0.1)
    return run_tabular_training(agent, policy, _RIVERSWIM, runner)


def bench_tabular(train_episodes: int = 400, max_episode_steps: int = 50) -> dict[str, Any]:
    runner = RunnerConfig(
        seed=0,
        train_episodes=train_episodes,
        train_steps=None,
        max_episode_steps=max_episode_steps,
        eval_episodes=0,
        checkpoint_freq=None,
        checkpoint_dir="checkpoints",
        save_final_checkpoint=False,
    )
    env_steps = train_episodes * max_episode_steps

    # First call pays the JIT compile cost; the second measures the cached run.
    start = time.perf_counter()
    result = _run(runner)
    np.asarray(result.q_table)  # force device sync
    first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    result = _run(runner)
    np.asarray(result.q_table)
    run_seconds = time.perf_counter() - start

    return {
        "env_steps": env_steps,
        "compile_seconds": round(max(first_seconds - run_seconds, 0.0), 4),
        "run_seconds": round(run_seconds, 4),
        "steps_per_second": round(env_steps / run_seconds, 1) if run_seconds > 0 else None,
    }


def run_benchmarks() -> dict[str, Any]:
    return {"tabular_q_learning_riverswim": bench_tabular()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Write the JSON report to this path.")
    args = parser.parse_args()

    report = run_benchmarks()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path

        Path(args.out).expanduser().write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
