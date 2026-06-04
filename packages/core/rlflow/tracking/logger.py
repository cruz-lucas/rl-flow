from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from rlflow.tracking.manifest import utc_now_iso


class LoggerBackend(Protocol):
    def log_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        episode: int | None = None,
        phase: str = "train",
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def log_artifact(
        self,
        path: Path,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class JsonlLogger:
    def __init__(self, run_dir: str | Path, path: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.path = Path(path) if path is not None else self.run_dir / "logs" / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        episode: int | None = None,
        phase: str = "train",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "type": "metric",
            "created_at": utc_now_iso(),
            "phase": phase,
            "name": name,
            "value": float(value),
        }
        if step is not None:
            row["step"] = int(step)
        if episode is not None:
            row["episode"] = int(episode)
        if metadata:
            row["metadata"] = metadata
        self._write(row)

    def log_artifact(
        self,
        path: Path,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact_path = Path(path)
        row: dict[str, Any] = {
            "type": "artifact",
            "created_at": utc_now_iso(),
            "name": name or artifact_path.name,
            "path": self._display_path(artifact_path),
        }
        if metadata:
            row["metadata"] = metadata
        self._write(row)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.run_dir.resolve()))
        except ValueError:
            return str(path)


class CompositeLogger:
    def __init__(self, backends: list[LoggerBackend]) -> None:
        self.backends = backends

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        episode: int | None = None,
        phase: str = "train",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for backend in self.backends:
            backend.log_metric(
                name,
                value,
                step=step,
                episode=episode,
                phase=phase,
                metadata=metadata,
            )

    def log_artifact(
        self,
        path: Path,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for backend in self.backends:
            backend.log_artifact(path, name=name, metadata=metadata)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()
