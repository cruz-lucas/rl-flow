from __future__ import annotations

import hashlib
import platform as platform_module
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rlflow.run.v1"] = "rlflow.run.v1"
    run_id: str
    experiment_id: str
    sweep_id: str | None = None
    sweep_group_id: str | None = None
    sweep_trial_id: str | None = None
    seed: Any | None = None
    created_at: str
    run_dir: str
    workflow_path: str
    resolved_config_path: str
    generated_gin_path: str
    command_path: str
    workflow_sha256: str
    resolved_config_sha256: str
    generated_gin_sha256: str
    command_sha256: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str
    platform: str
    dependencies: dict[str, str] = Field(default_factory=dict)
    backend: Literal["local", "slurm"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    path = Path(run_dir) / "manifest.json"
    _atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")
    return path


def load_manifest(run_dir_or_path: str | Path) -> RunManifest:
    path = Path(run_dir_or_path)
    if path.is_dir():
        path = path / "manifest.json"
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def collect_git_info(cwd: str | Path) -> tuple[str | None, bool | None]:
    root = Path(cwd)
    commit = _git(["rev-parse", "HEAD"], root)
    dirty_text = _git(["status", "--porcelain"], root)
    dirty = None if dirty_text is None else bool(dirty_text.strip())
    return commit, dirty


def collect_dependency_versions() -> dict[str, str]:
    names = [
        "rl-flow",
        "fastapi",
        "pydantic",
        "pydantic-settings",
        "sqlmodel",
        "jinja2",
        "typer",
        "PyYAML",
        "jsonschema",
        "jax",
        "optax",
        "navix",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def python_version() -> str:
    return sys.version.replace("\n", " ")


def platform_string() -> str:
    return platform_module.platform()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
