from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rlflow.schemas.workflow import WorkflowSpec


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    workflow: WorkflowSpec
    resolved_config: dict[str, Any]
    run_dir: str
    command: str
    generated_files: list[str] = Field(default_factory=list)
    execution_backend: Literal["local", "slurm"]
