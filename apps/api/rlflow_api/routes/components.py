from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from rlflow.schemas.component import ComponentSpec

router = APIRouter(prefix="/components", tags=["components"])


@router.get("", response_model=list[ComponentSpec])
def list_components(request: Request) -> list[ComponentSpec]:
    return request.app.state.registry.list()


@router.get("/{component_id}", response_model=ComponentSpec)
def get_component(component_id: str, request: Request) -> ComponentSpec:
    component = request.app.state.registry.maybe_get(component_id)
    if component is None:
        raise HTTPException(status_code=404, detail=f"Unknown component: {component_id}")
    return component
