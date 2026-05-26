from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentRecord(SQLModel, table=True):
    experiment_id: str = Field(primary_key=True)
    workflow_spec: dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    resolved_config: dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    run_dir: str
    command: str
    generated_files: list[str] = Field(sa_column=Column(JSON), default_factory=list)
    execution_backend: str
    status: str = "compiled"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class WorkflowRecord(SQLModel, table=True):
    workflow_id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    workflow_spec: dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class JobRecord(SQLModel, table=True):
    job_id: str = Field(primary_key=True)
    experiment_id: str = Field(index=True)
    backend: str
    status: str
    message: str = ""
    run_dir: str
    external_id: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
