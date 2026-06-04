from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.compiler import WorkflowCompilationError, WorkflowCompiler
from rlflow.graph.run_naming import make_flow_run_dir, make_run_id
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo
from rlflow.schemas.sweep import SweepCompilation, SweepTrial
from rlflow.schemas.workflow import WorkflowSpec
from rlflow.storage.models import ExperimentRecord
from rlflow.tracking.status import RunStatusState, load_status

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
    results: list[dict[str, Any]] = []
    seen_run_dirs: set[Path] = set()
    for record in request.app.state.storage.list_experiments():
        result = _experiment_result(record)
        results.append(result)
        seen_run_dirs.add(Path(result["run_dir"]).expanduser().resolve())

    for result in _filesystem_sweep_results(_absolute_run_root(request)):
        run_dir = Path(result["run_dir"]).expanduser().resolve()
        if run_dir in seen_run_dirs:
            continue
        results.append(result)
        seen_run_dirs.add(run_dir)

    results.sort(key=_result_modified_time, reverse=True)
    return results


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
    return _result_from_workflow(
        experiment_id=record.experiment_id,
        status=_record_status(run_dir, record.status),
        run_dir=run_dir,
        workflow=workflow,
        metadata=metadata,
        sweep_dir=metadata.get("sweep_dir") if isinstance(metadata, dict) else None,
    )


def _filesystem_sweep_results(run_root: Path) -> list[dict[str, Any]]:
    if not run_root.exists():
        return []
    results: list[dict[str, Any]] = []
    for manifest_path in run_root.rglob("sweep_manifest.yaml"):
        try:
            manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            compilation = SweepCompilation.model_validate(manifest_data)
        except Exception:
            continue
        for trial in compilation.trials:
            results.append(_sweep_trial_result(compilation, trial))
    return results


def _sweep_trial_result(compilation: SweepCompilation, trial: SweepTrial) -> dict[str, Any]:
    workflow = _read_yaml_dict(Path(trial.workflow_path))
    metadata = workflow.get("metadata", {}) if isinstance(workflow, dict) else {}
    metadata = {
        "experiment_id": trial.experiment_id,
        "sweep_id": compilation.sweep_id,
        "sweep_trial_id": trial.trial_id,
        "sweep_group_id": trial.group_id,
        "sweep_group_run_dir": trial.group_run_dir,
        "sweep_parameters": trial.parameters,
        "sweep_group_parameters": _non_seed_parameters(trial.parameters),
        "seed": trial.seed_value,
        "sweep_dir": compilation.sweep_dir,
        **(metadata if isinstance(metadata, dict) else {}),
    }
    run_dir = Path(trial.run_dir)
    return _result_from_workflow(
        experiment_id=str(metadata.get("experiment_id") or trial.experiment_id),
        status=_filesystem_status(run_dir),
        run_dir=run_dir,
        workflow=workflow,
        metadata=metadata,
        sweep_dir=compilation.sweep_dir,
    )


def _result_from_workflow(
    *,
    experiment_id: str,
    status: str,
    run_dir: Path,
    workflow: dict[str, Any],
    metadata: dict[str, Any],
    sweep_dir: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "run_dir": str(run_dir),
        "workflow_name": (
            workflow.get("name", experiment_id)
            if isinstance(workflow, dict)
            else experiment_id
        ),
        "sweep_dir": sweep_dir,
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
        "metrics": _read_metrics(run_dir),
        "train_history": _read_jsonl(run_dir / "logs" / "train_history.jsonl"),
        "eval_history": _read_jsonl(run_dir / "logs" / "eval_history.jsonl"),
    }


def _filesystem_status(run_dir: Path) -> str:
    status = load_status(run_dir)
    if status is not None:
        return status.status.value
    if (
        (run_dir / "summaries" / "metrics.json").exists()
        or (run_dir / "metrics.json").exists()
    ):
        return RunStatusState.completed.value
    if (
        (run_dir / "logs" / "train_history.jsonl").exists()
        or (run_dir / "logs" / "eval_history.jsonl").exists()
    ):
        return RunStatusState.completed.value
    return RunStatusState.compiled.value


def _record_status(run_dir: Path, fallback: str) -> str:
    status = load_status(run_dir)
    if status is not None:
        return status.status.value
    if (
        (run_dir / "summaries" / "metrics.json").exists()
        or (run_dir / "metrics.json").exists()
        or (run_dir / "logs" / "train_history.jsonl").exists()
        or (run_dir / "logs" / "eval_history.jsonl").exists()
    ):
        return RunStatusState.completed.value
    return fallback


def _result_modified_time(result: dict[str, Any]) -> float:
    run_dir = Path(str(result.get("run_dir", ""))).expanduser()
    candidates = [
        run_dir / "status.json",
        run_dir / "summaries" / "metrics.json",
        run_dir / "metrics.json",
        run_dir / "logs" / "train_history.jsonl",
        run_dir / "logs" / "eval_history.jsonl",
        run_dir / "workflow.yaml",
        run_dir,
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def _absolute_run_root(request: Request) -> Path:
    run_root = Path(request.app.state.settings.run_root).expanduser()
    if run_root.is_absolute():
        return run_root.resolve()
    return (Path.cwd() / run_root).resolve()


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _non_seed_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if not _is_seed_parameter(key)
    }


def _is_seed_parameter(key: str) -> bool:
    normalized = key.lower()
    return normalized == "seed" or normalized.endswith("_seed") or normalized.endswith(".seed")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _read_json(run_dir / "summaries" / "metrics.json")
    if metrics:
        return metrics
    return _read_json(run_dir / "metrics.json")


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
