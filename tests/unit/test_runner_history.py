import json
from pathlib import Path

import numpy as np

from rlflow_builtin.runners.tabular_jax import _write_episode_history, _write_eval_history


def test_history_writers_include_cumulative_env_step(tmp_path: Path) -> None:
    train_path = tmp_path / "train_history.jsonl"
    eval_path = tmp_path / "eval_history.jsonl"

    _write_episode_history(
        train_path,
        returns=np.array([1.0, 2.0]),
        lengths=np.array([3, 4]),
        losses=np.array([0.1, 0.2]),
    )
    _write_eval_history(
        eval_path,
        returns=np.array([5.0, 6.0]),
        lengths=np.array([2, 5]),
    )

    train_rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [row["env_step"] for row in train_rows] == [3, 7]
    assert [row["env_step"] for row in eval_rows] == [2, 7]
