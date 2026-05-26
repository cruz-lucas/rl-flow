from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class JobState(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    unknown = "unknown"


class JobStatus(BaseModel):
    state: JobState
    message: str = ""


class JobInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    experiment_id: str
    backend: str
    status: JobStatus = Field(default_factory=lambda: JobStatus(state=JobState.pending))
    run_dir: str
    external_id: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
