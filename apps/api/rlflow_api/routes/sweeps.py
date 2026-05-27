from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.run_naming import make_run_id, slugify_run_name
from rlflow.graph.sweep import SweepCompilationError, SweepCompiler
from rlflow.schemas.job import JobInfo
from rlflow.schemas.sweep import SweepCompilation, SweepMetric, SweepParameter, SweepSlurmSpec, SweepSpec
from rlflow.schemas.workflow import ExecutionSpec, WorkflowSpec

router = APIRouter(prefix="/sweeps", tags=["sweeps"])


class SweepCandidate(BaseModel):
    target: str
    label: str
    node_id: str
    component: str
    component_display_name: str
    field: str
    value: Any
    value_type: str
    recommended_values: list[Any] = Field(default_factory=list)


class SweepCandidateResponse(BaseModel):
    workflow_id: str
    workflow_name: str
    candidates: list[SweepCandidate]
    seed_candidates: list[SweepCandidate]


class SweepListItem(BaseModel):
    path: str
    sweep_id: str
    name: str
    trial_count: int
    modified_time: float


class SweepInspectRequest(BaseModel):
    path: str
    metric_name: str = "mean_eval_return"
    metric_goal: Literal["maximize", "minimize"] = "maximize"
    metric_last_n: int | None = Field(default=None, ge=1)


class SweepParameterRequest(BaseModel):
    label: str
    target: str
    values: list[Any] | None = None
    distribution: Literal["choice", "uniform", "loguniform", "int_uniform"] = "choice"
    minimum: float | int | None = None
    maximum: float | int | None = None


class SweepBuildRequest(BaseModel):
    workflow_id: str
    name: str | None = None
    description: str = ""
    method: Literal["grid", "random"] = "grid"
    metric_name: str = "mean_eval_return"
    metric_goal: Literal["maximize", "minimize"] = "maximize"
    metric_last_n: int | None = Field(default=None, ge=1)
    execution_backend: Literal["local", "slurm"] | None = None
    parameters: list[SweepParameterRequest]
    seed_target: str | None = None
    seed_start: int = Field(default=0, ge=0)
    seed_count: int = Field(default=0, ge=0, le=1000)
    num_trials: int | None = Field(default=None, ge=1)
    random_seed: int = Field(default=0, ge=0)
    slurm_max_parallel: int | None = Field(default=None, ge=1)


class SweepRunResponse(BaseModel):
    compilation: SweepCompilation
    jobs: list[JobInfo]


@router.get("", response_model=list[SweepListItem])
def list_sweeps(request: Request) -> list[SweepListItem]:
    run_root = _absolute_run_root(request)
    if not run_root.exists():
        return []
    sweeps: list[SweepListItem] = []
    for path in run_root.rglob("sweep_manifest.yaml"):
        try:
            compilation = SweepCompilation.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        sweeps.append(
            SweepListItem(
                path=_display_path(path),
                sweep_id=compilation.sweep_id,
                name=compilation.name,
                trial_count=len(compilation.trials),
                modified_time=path.stat().st_mtime,
            )
        )
    sweeps.sort(key=lambda item: item.modified_time, reverse=True)
    return sweeps


@router.post("/inspect")
def inspect_sweep(payload: SweepInspectRequest, request: Request) -> dict[str, Any]:
    manifest_path = _resolve_sweep_manifest_path(payload.path, request)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Sweep manifest does not exist: {manifest_path}")
    try:
        return SweepCompiler(request.app.state.registry).summarize(
            manifest_path,
            metric=payload.metric_name,
            goal=payload.metric_goal,
            metric_last_n=payload.metric_last_n,
        )
    except (SweepCompilationError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workflows/{workflow_id}/candidates", response_model=SweepCandidateResponse)
def workflow_sweep_candidates(workflow_id: str, request: Request) -> SweepCandidateResponse:
    workflow = _saved_workflow(workflow_id, request)
    candidates = _workflow_candidates(workflow, request)
    return SweepCandidateResponse(
        workflow_id=workflow_id,
        workflow_name=workflow.name,
        candidates=candidates,
        seed_candidates=[
            candidate
            for candidate in candidates
            if candidate.field == "seed" or candidate.field.endswith("_seed")
        ],
    )


@router.post("/compile", response_model=SweepCompilation)
def compile_sweep(payload: SweepBuildRequest, request: Request) -> SweepCompilation:
    compilation, _experiments = _compile_payload(payload, request)
    return compilation


@router.post("/run", response_model=SweepRunResponse)
def run_sweep(payload: SweepBuildRequest, request: Request) -> SweepRunResponse:
    compilation, experiments = _compile_payload(payload, request)
    backend = experiments[0].execution_backend if experiments else payload.execution_backend
    if backend == "slurm":
        if compilation.slurm_array_path is None:
            raise HTTPException(status_code=422, detail="Sweep did not generate a SLURM array script")
        job = request.app.state.slurm_executor.submit_array(compilation)
        request.app.state.storage.save_job(job)
        return SweepRunResponse(compilation=compilation, jobs=[job])
    if backend == "local":
        jobs: list[JobInfo] = []
        for experiment in experiments:
            job = request.app.state.local_executor.submit(experiment)
            request.app.state.storage.save_job(job)
            request.app.state.storage.save_experiment(experiment, status="running")
            jobs.append(job)
        return SweepRunResponse(compilation=compilation, jobs=jobs)
    raise HTTPException(status_code=422, detail=f"Unsupported sweep backend: {backend}")


def _compile_payload(payload: SweepBuildRequest, request: Request):
    workflow = _saved_workflow(payload.workflow_id, request)
    if payload.execution_backend is not None:
        workflow.execution.backend = payload.execution_backend
    spec = _sweep_spec(payload, workflow)
    sweep_id = spec.sweep_id or make_run_id(spec.name)
    spec.sweep_id = sweep_id
    sweep_dir = (
        _absolute_run_root(request)
        / "sweeps"
        / slugify_run_name(spec.name)
        / sweep_id
    )
    try:
        compilation = SweepCompiler(request.app.state.registry).compile(spec, out_dir=sweep_dir)
    except (SweepCompilationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    experiments = []
    for trial in compilation.trials:
        trial_workflow = WorkflowSpec.model_validate(
            yaml.safe_load(Path(trial.workflow_path).read_text(encoding="utf-8"))
        )
        experiment = SweepCompiler(request.app.state.registry).workflow_compiler.compile(
            trial_workflow,
            out_dir=trial.run_dir,
        )
        request.app.state.storage.save_experiment(experiment, status="compiled")
        experiments.append(experiment)
    return compilation, experiments


def _sweep_spec(payload: SweepBuildRequest, workflow: WorkflowSpec) -> SweepSpec:
    parameters: dict[str, SweepParameter] = {}
    for item in payload.parameters:
        parameters[item.label or item.target] = SweepParameter(
            target=item.target,
            values=item.values,
            distribution=item.distribution,
            minimum=item.minimum,
            maximum=item.maximum,
        )
    if payload.seed_count > 0:
        if not payload.seed_target:
            raise HTTPException(status_code=422, detail="Select a seed target or set seed count to zero")
        parameters["seed"] = SweepParameter(
            target=payload.seed_target,
            values=list(range(payload.seed_start, payload.seed_start + payload.seed_count)),
        )
    if not parameters:
        raise HTTPException(status_code=422, detail="Select at least one sweep parameter or seed")

    execution = None
    if payload.execution_backend is not None:
        execution = ExecutionSpec(
            backend=payload.execution_backend,
            cluster=workflow.execution.cluster,
            options=workflow.execution.options,
        )
    return SweepSpec(
        name=(payload.name or f"{workflow.name} sweep"),
        description=payload.description,
        sweep_id=None,
        workflow=workflow,
        method=payload.method,
        metric=SweepMetric(name=payload.metric_name, goal=payload.metric_goal, last_n=payload.metric_last_n),
        parameters=parameters,
        num_trials=payload.num_trials,
        seed=payload.random_seed,
        execution=execution,
        slurm=SweepSlurmSpec(max_parallel=payload.slurm_max_parallel),
    )


def _saved_workflow(workflow_id: str, request: Request) -> WorkflowSpec:
    record = request.app.state.storage.get_workflow(workflow_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown saved workflow: {workflow_id}")
    return WorkflowSpec.model_validate(record.workflow_spec).model_copy(deep=True)


def _workflow_candidates(workflow: WorkflowSpec, request: Request) -> list[SweepCandidate]:
    candidates: list[SweepCandidate] = []
    for node in workflow.nodes:
        component = request.app.state.registry.get(node.component)
        properties = component.config_schema.get("properties", {})
        for field, schema in properties.items():
            if not isinstance(schema, dict) or schema.get("x-inspector-hidden") or schema.get("deprecated"):
                continue
            value_type = _candidate_type(schema)
            if value_type is None:
                continue
            value = node.config.get(field, component.defaults.get(field))
            candidates.append(
                SweepCandidate(
                    target=f"nodes.{node.id}.config.{field}",
                    label=f"{node.id}.{field}",
                    node_id=node.id,
                    component=node.component,
                    component_display_name=component.display_name,
                    field=field,
                    value=value,
                    value_type=value_type,
                    recommended_values=_recommended_values(field, value, schema, value_type),
                )
            )
    candidates.sort(key=lambda item: (item.node_id, item.field))
    return candidates


def _candidate_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    types = set(schema_type if isinstance(schema_type, list) else [schema_type])
    types.discard(None)
    if "integer" in types:
        return "integer"
    if "number" in types:
        return "number"
    if "boolean" in types:
        return "boolean"
    if "string" in types and schema.get("enum"):
        return "choice"
    return None


def _recommended_values(field: str, value: Any, schema: dict[str, Any], value_type: str) -> list[Any]:
    if field == "seed" or field.endswith("_seed"):
        return [0, 1, 2]
    if value_type == "boolean":
        return [True, False]
    if value_type == "choice":
        enum_values = [item for item in schema.get("enum", []) if item is not None]
        if value in enum_values:
            return [value, *[item for item in enum_values if item != value][:2]]
        return enum_values[:3]
    if value_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
        minimum = int(schema.get("minimum", 1))
        return _unique([max(minimum, value // 2), value, max(minimum, value * 2)])
    if value_type == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number > 0 and "learning_rate" in field:
            return _unique([number, number * 0.3, number * 0.1])
        if number > 0:
            return _unique([number / 2, number, number * 2])
        return _unique([number, number + 0.1, number + 1.0])
    return []


def _unique(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _absolute_run_root(request: Request) -> Path:
    run_root = Path(request.app.state.settings.run_root).expanduser()
    if run_root.is_absolute():
        return run_root.resolve()
    return (Path.cwd() / run_root).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_sweep_manifest_path(path: str, request: Request) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
        if candidate.is_dir() or candidate.suffix == "":
            manifest = (candidate / "sweep_manifest.yaml").resolve()
            if manifest not in candidates:
                candidates.append(manifest)

    if raw_path.is_absolute():
        add_candidate(raw_path)
    else:
        cwd = Path.cwd()
        run_root = _absolute_run_root(request)
        add_candidate(cwd / raw_path)
        if raw_path.parts and raw_path.parts[0] == run_root.name:
            add_candidate(run_root.parent / raw_path)
        else:
            add_candidate(run_root / raw_path)

    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                return candidate / "sweep_manifest.yaml"
            return candidate
    return candidates[0] if candidates else raw_path.resolve()
