from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rlflow.schemas.sweep import SweepCompilation, SweepTrial


def load_sweep_manifest(path: str | Path) -> SweepCompilation:
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    path = path.resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    compilation = SweepCompilation.model_validate(data)
    return _rebase_compilation_paths(compilation, manifest_path=path)


def _rebase_compilation_paths(
    compilation: SweepCompilation,
    *,
    manifest_path: Path,
) -> SweepCompilation:
    recorded_sweep_dir = Path(compilation.sweep_dir).expanduser().resolve()
    sweep_dir = manifest_path.parent

    def rebase(value: str | None) -> str | None:
        if not value:
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            return str(sweep_dir / path)
        try:
            relative = path.relative_to(recorded_sweep_dir)
        except ValueError:
            return value
        return str(sweep_dir / relative)

    trials = [
        trial.model_copy(
            update={
                "group_run_dir": rebase(trial.group_run_dir),
                "run_dir": rebase(trial.run_dir),
                "command": rebase(trial.command),
                "workflow_path": rebase(trial.workflow_path),
                "metrics_path": rebase(trial.metrics_path),
            }
        )
        for trial in compilation.trials
    ]
    return compilation.model_copy(
        update={
            "sweep_dir": str(sweep_dir),
            "manifest_path": str(manifest_path),
            "slurm_array_path": rebase(compilation.slurm_array_path),
            "trials": trials,
            "generated_files": [rebase(path) for path in compilation.generated_files],
        }
    )


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
