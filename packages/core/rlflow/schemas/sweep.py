from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlflow.schemas.workflow import ExecutionSpec, WorkflowSpec


SweepMethod = Literal["grid", "random"]
SweepGoal = Literal["maximize", "minimize"]
SweepDistribution = Literal["choice", "uniform", "loguniform", "int_uniform"]


class SweepMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "mean_eval_return"
    goal: SweepGoal = "maximize"
    last_n: int | None = Field(default=None, ge=1)


class SweepParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    values: list[Any] | None = None
    distribution: SweepDistribution = "choice"
    minimum: float | int | None = None
    maximum: float | int | None = None

    @model_validator(mode="after")
    def validate_search_space(self) -> "SweepParameter":
        if self.values is not None:
            if not self.values:
                raise ValueError("values must not be empty")
            return self
        if self.distribution == "choice":
            raise ValueError("choice parameters require values")
        if self.minimum is None or self.maximum is None:
            raise ValueError(f"{self.distribution} parameters require minimum and maximum")
        if self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        if self.distribution == "loguniform" and self.minimum <= 0:
            raise ValueError("loguniform minimum must be greater than 0")
        return self


class SweepSlurmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_parallel: int | None = Field(default=None, ge=1)


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    sweep_id: str | None = None
    workflow: str | WorkflowSpec
    method: SweepMethod = "grid"
    metric: SweepMetric = Field(default_factory=SweepMetric)
    parameters: dict[str, SweepParameter]
    num_trials: int | None = Field(default=None, ge=1)
    seed: int = 0
    execution: ExecutionSpec | None = None
    slurm: SweepSlurmSpec = Field(default_factory=SweepSlurmSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trial_count(self) -> "SweepSpec":
        if not self.parameters:
            raise ValueError("sweep parameters must not be empty")
        if self.method == "random" and self.num_trials is None:
            self.num_trials = 20
        return self


class SweepTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    trial_id: str
    experiment_id: str
    parameters: dict[str, Any]
    run_dir: str
    command: str
    workflow_path: str
    metrics_path: str


class SweepCompilation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sweep_id: str
    name: str
    method: SweepMethod
    metric: SweepMetric
    sweep_dir: str
    manifest_path: str
    slurm_array_path: str | None = None
    trials: list[SweepTrial]
    generated_files: list[str] = Field(default_factory=list)
