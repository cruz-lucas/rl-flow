"""Shared helpers for the FastAPI route modules.

These were previously copy-pasted across ``experiments``, ``jobs``, ``sweeps``,
and ``datasets``; they live here so there is a single implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, Request

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor


def executor(request: Request, backend: str) -> LocalExecutor | SlurmExecutor:
    if backend == "local":
        return request.app.state.local_executor
    if backend == "slurm":
        return request.app.state.slurm_executor
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")


def absolute_run_root(request: Request) -> Path:
    run_root = Path(request.app.state.settings.run_root).expanduser()
    if run_root.is_absolute():
        return run_root.resolve()
    return (Path.cwd() / run_root).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
