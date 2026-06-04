from __future__ import annotations

import socket
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunStatusState(str, Enum):
    compiled = "compiled"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    unknown = "unknown"


class RunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rlflow.status.v1"] = "rlflow.status.v1"
    status: RunStatusState
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    message: str | None = None
    hostname: str | None = None
    backend: str | None = None
    external_id: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_status(run_dir_or_path: str | Path) -> RunStatus | None:
    path = Path(run_dir_or_path)
    if path.is_dir():
        path = path / "status.json"
    if not path.exists():
        return None
    try:
        return RunStatus.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_status(run_dir: str | Path, status: RunStatus) -> Path:
    path = Path(run_dir) / "status.json"
    _atomic_write_text(path, status.model_dump_json(indent=2) + "\n")
    return path


def update_status(
    run_dir: str | Path,
    state: RunStatusState | str,
    *,
    exit_code: int | None = None,
    message: str | None = None,
    backend: str | None = None,
    external_id: str | None = None,
    hostname: str | None = None,
) -> RunStatus:
    run_dir = Path(run_dir)
    previous = load_status(run_dir)
    now = utc_now_iso()
    state = RunStatusState(state)
    clean_external_id = external_id or None
    clean_backend = backend or None
    host = hostname
    if host is None and previous is not None:
        host = previous.hostname
    if state in {RunStatusState.running, RunStatusState.completed, RunStatusState.failed, RunStatusState.cancelled}:
        host = host or socket.gethostname()

    started_at = previous.started_at if previous is not None else None
    if state == RunStatusState.running and started_at is None:
        started_at = now
    if state in {RunStatusState.completed, RunStatusState.failed, RunStatusState.cancelled} and started_at is None:
        started_at = now

    finished_at = previous.finished_at if previous is not None else None
    if is_terminal_status(state):
        finished_at = now
    elif state in {RunStatusState.compiled, RunStatusState.queued, RunStatusState.running}:
        finished_at = None

    status = RunStatus(
        status=state,
        created_at=previous.created_at if previous is not None else now,
        updated_at=now,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code if exit_code is not None else (previous.exit_code if previous is not None else None),
        message=message if message is not None else (previous.message if previous is not None else None),
        hostname=host,
        backend=clean_backend or (previous.backend if previous is not None else None),
        external_id=clean_external_id or (previous.external_id if previous is not None else None),
    )
    write_status(run_dir, status)
    return status


def is_terminal_status(state: RunStatusState | str) -> bool:
    state = RunStatusState(state)
    return state in {
        RunStatusState.completed,
        RunStatusState.failed,
        RunStatusState.cancelled,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
