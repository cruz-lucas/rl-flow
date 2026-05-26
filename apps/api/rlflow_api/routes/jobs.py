from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from rlflow.execution.local import LocalExecutor
from rlflow.schemas.job import JobInfo, JobState, JobStatus
from rlflow.storage.models import JobRecord

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobInfo])
def list_jobs(request: Request) -> list[JobInfo]:
    return [_refresh_job(request, record) for record in request.app.state.storage.list_jobs()]


@router.get("/{job_id}", response_model=JobInfo)
def get_job(job_id: str, request: Request) -> JobInfo:
    record = request.app.state.storage.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return _refresh_job(request, record)


@router.post("/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str, request: Request) -> JobStatus:
    record = request.app.state.storage.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    _executor(request, record.backend).cancel(job_id)
    status = _executor(request, record.backend).status(job_id)
    request.app.state.storage.update_job_status(job_id, status)
    return status


@router.get("/{job_id}/logs", response_model=str)
def job_logs(job_id: str, request: Request) -> str:
    record = request.app.state.storage.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    logs = _executor(request, record.backend).logs(job_id)
    if logs:
        return logs
    return "\n".join(
        path
        for path in [record.stdout_path, record.stderr_path]
        if path
    )


def _refresh_job(request: Request, record: JobRecord) -> JobInfo:
    info = request.app.state.storage.job_info_from_record(record)
    if info.status.state not in {JobState.pending, JobState.running}:
        return info

    executor = _executor(request, record.backend)
    if isinstance(executor, LocalExecutor):
        status = executor.status_for_job(info)
    else:
        status = executor.status(record.job_id)

    updated = request.app.state.storage.update_job_status(record.job_id, status)
    if updated is not None:
        info = request.app.state.storage.job_info_from_record(updated)
    else:
        info.status = status

    if status.state in {JobState.succeeded, JobState.failed, JobState.cancelled, JobState.unknown}:
        request.app.state.storage.update_experiment_status(info.experiment_id, status.state.value)
    return info


def _executor(request: Request, backend: str):
    if backend == "local":
        return request.app.state.local_executor
    if backend == "slurm":
        return request.app.state.slurm_executor
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
