from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rlflow.schemas.sweep import SweepCompilation, SweepTrial


def load_sweep_manifest(path: str | Path) -> SweepCompilation:
    path = Path(path)
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SweepCompilation.model_validate(data)


def load_trial_history(
    trial: SweepTrial,
    *,
    history: str = "train",
) -> pd.DataFrame:
    if history not in {"train", "eval"}:
        raise ValueError("history must be 'train' or 'eval'")

    path = Path(trial.run_dir) / "logs" / f"{history}_history.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row["trial_id"] = trial.trial_id
        row["experiment_id"] = trial.experiment_id
        row["run_dir"] = trial.run_dir
        row["group_id"] = trial.group_id
        row["seed_value"] = trial.seed_value
        row["parameters"] = trial.parameters
        rows.append(row)

    return pd.DataFrame(rows)


def load_histories(
    sweep_dir: str | Path,
    history: str = "train",
) -> pd.DataFrame:
    compilation = load_sweep_manifest(sweep_dir)
    frames = [
        load_trial_history(trial, history=history)
        for trial in compilation.trials
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_history_columns())

    df = pd.concat(frames, ignore_index=True)
    for column in _history_columns():
        if column not in df.columns:
            df[column] = pd.NA
    return df


def load_sweep_histories(
    manifest_path: str | Path,
    *,
    history: str = "train",
) -> pd.DataFrame:
    return load_histories(manifest_path, history=history)


def _history_columns() -> list[str]:
    return [
        "trial_id",
        "group_id",
        "seed_value",
        "run_dir",
        "episode",
        "env_step",
        "return",
        "discounted_return",
        "length",
        "loss",
        "parameters",
    ]


def is_seed_parameter(key: str) -> bool:
    normalized = key.lower()
    return normalized == "seed" or normalized.endswith("_seed") or normalized.endswith(".seed")


def non_seed_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if not is_seed_parameter(key)
    }
