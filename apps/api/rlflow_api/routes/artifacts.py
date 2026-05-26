from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/{experiment_id}", response_model=list[str])
def list_artifacts(experiment_id: str, request: Request) -> list[str]:
    record = request.app.state.storage.get_experiment(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_id}")
    return request.app.state.artifacts.list(record.run_dir)
