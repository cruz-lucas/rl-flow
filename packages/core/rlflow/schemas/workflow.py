from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: Literal["local", "slurm"] = "local"
    cluster: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    component: str
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_node: str
    from_port: str
    to_node: str
    to_port: str


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationErrorDetail(BaseModel):
    message: str
    node_id: str | None = None
    field: str | None = None
    code: str = "validation_error"


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationErrorDetail] = Field(default_factory=list)
