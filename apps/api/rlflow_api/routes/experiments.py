from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.compiler import WorkflowCompilationError, WorkflowCompiler
from rlflow.graph.run_naming import make_flow_run_dir, make_run_id
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo
from rlflow.schemas.workflow import WorkflowSpec
from rlflow.storage.models import ExperimentRecord

router = APIRouter(prefix="/experiments", tags=["experiments"])


class RunExperimentRequest(BaseModel):
    workflow: WorkflowSpec | None = None
    experiment: ExperimentSpec | None = None
    backend: Literal["local", "slurm"] | None = None
    out_dir: str | None = None


@router.get("", response_model=list[ExperimentRecord])
def list_experiments(request: Request) -> list[ExperimentRecord]:
    return request.app.state.storage.list_experiments()


@router.get("/results")
def list_experiment_results(request: Request) -> list[dict[str, Any]]:
    return [_experiment_result(record) for record in request.app.state.storage.list_experiments()]


@router.get("/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(experiment_id: str, request: Request) -> ExperimentRecord:
    record = request.app.state.storage.get_experiment(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_id}")
    return record


@router.post("/run", response_model=JobInfo)
def run_experiment(payload: RunExperimentRequest, request: Request) -> JobInfo:
    experiment = payload.experiment
    if payload.workflow is not None:
        workflow = payload.workflow.model_copy(deep=True)
        if payload.backend is not None:
            workflow.execution.backend = payload.backend
        if payload.out_dir:
            out_dir = Path(payload.out_dir)
        else:
            run_id = make_run_id(workflow.name)
            workflow.metadata = {**workflow.metadata, "experiment_id": run_id}
            out_dir = make_flow_run_dir(
                request.app.state.settings.run_root,
                workflow.name,
                run_id,
            )
        try:
            experiment = WorkflowCompiler(request.app.state.registry).compile(workflow, out_dir=out_dir)
        except WorkflowCompilationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request.app.state.storage.save_experiment(experiment, status="compiled")

    if experiment is None:
        raise HTTPException(status_code=400, detail="Provide either workflow or experiment")

    backend = payload.backend or experiment.execution_backend
    executor = _executor(request, backend)
    job = executor.submit(experiment)
    request.app.state.storage.save_job(job)
    request.app.state.storage.save_experiment(experiment, status="running")
    return job


def _executor(request: Request, backend: str) -> LocalExecutor | SlurmExecutor:
    if backend == "local":
        return request.app.state.local_executor
    if backend == "slurm":
        return request.app.state.slurm_executor
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")


def _experiment_result(record: ExperimentRecord) -> dict[str, Any]:
    run_dir = Path(record.run_dir)
    workflow = record.workflow_spec or {}
    metadata = workflow.get("metadata", {}) if isinstance(workflow, dict) else {}
    return {
        "experiment_id": record.experiment_id,
        "status": record.status,
        "run_dir": record.run_dir,
        "workflow_name": (
            workflow.get("name", record.experiment_id)
            if isinstance(workflow, dict)
            else record.experiment_id
        ),
        "sweep_id": metadata.get("sweep_id") if isinstance(metadata, dict) else None,
        "sweep_trial_id": metadata.get("sweep_trial_id") if isinstance(metadata, dict) else None,
        "sweep_group_id": metadata.get("sweep_group_id") if isinstance(metadata, dict) else None,
        "sweep_group_run_dir": metadata.get("sweep_group_run_dir") if isinstance(metadata, dict) else None,
        "sweep_parameters": metadata.get("sweep_parameters", {}) if isinstance(metadata, dict) else {},
        "sweep_group_parameters": (
            metadata.get("sweep_group_parameters", {})
            if isinstance(metadata, dict)
            else {}
        ),
        "seed": metadata.get("seed") if isinstance(metadata, dict) else None,
        "metrics": _read_json(run_dir / "metrics.json"),
        "train_history": _read_jsonl(run_dir / "logs" / "train_history.jsonl"),
        "eval_history": _read_jsonl(run_dir / "logs" / "eval_history.jsonl"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
