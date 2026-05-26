from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rlflow.graph.compiler import WorkflowCompilationError, WorkflowCompiler
from rlflow.graph.run_naming import make_flow_run_dir, make_run_id
from rlflow.graph.validation import WorkflowValidator
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.workflow import ValidationResult, WorkflowSpec

router = APIRouter(prefix="/workflows", tags=["workflows"])


class CompileRequest(BaseModel):
    workflow: WorkflowSpec
    out_dir: str | None = None


class SaveWorkflowRequest(BaseModel):
    workflow: WorkflowSpec
    workflow_id: str | None = None


class WorkflowGalleryItem(BaseModel):
    workflow_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


def _gallery_item(record: object) -> WorkflowGalleryItem:
    return WorkflowGalleryItem.model_validate(record, from_attributes=True)


@router.get("", response_model=list[WorkflowGalleryItem])
def list_saved_workflows(request: Request) -> list[WorkflowGalleryItem]:
    return [_gallery_item(record) for record in request.app.state.storage.list_workflows()]


@router.post("", response_model=WorkflowGalleryItem)
def save_workflow(payload: SaveWorkflowRequest, request: Request) -> WorkflowGalleryItem:
    record = request.app.state.storage.save_workflow(payload.workflow, payload.workflow_id)
    return _gallery_item(record)


@router.post("/validate", response_model=ValidationResult)
def validate_workflow(workflow: WorkflowSpec, request: Request) -> ValidationResult:
    return WorkflowValidator(request.app.state.registry).validate(workflow)


@router.get("/examples/{name}", response_model=WorkflowSpec)
def get_example_workflow(name: str) -> WorkflowSpec:
    path = Path("configs/workflows") / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown example workflow: {name}")
    import yaml

    return WorkflowSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@router.get("/{workflow_id}", response_model=WorkflowSpec)
def get_saved_workflow(workflow_id: str, request: Request) -> WorkflowSpec:
    record = request.app.state.storage.get_workflow(workflow_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown saved workflow: {workflow_id}")
    return WorkflowSpec.model_validate(record.workflow_spec)


@router.post("/compile", response_model=ExperimentSpec)
def compile_workflow(payload: CompileRequest, request: Request) -> ExperimentSpec:
    compiler = WorkflowCompiler(request.app.state.registry)
    workflow = payload.workflow.model_copy(deep=True)
    if payload.out_dir:
        out_dir = Path(payload.out_dir)
    else:
        run_id = str(workflow.metadata.get("experiment_id") or make_run_id(workflow.name))
        workflow.metadata = {**workflow.metadata, "experiment_id": run_id}
        out_dir = make_flow_run_dir(
            request.app.state.settings.run_root,
            workflow.name,
            run_id,
        )
    try:
        experiment = compiler.compile(workflow, out_dir=out_dir)
    except WorkflowCompilationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request.app.state.storage.save_experiment(experiment)
    return experiment
