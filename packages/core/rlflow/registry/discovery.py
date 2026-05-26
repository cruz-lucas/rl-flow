from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from typing import Any

from rlflow.registry.base import ComponentRegistry
from rlflow.schemas.component import ComponentSpec


def _as_components(value: Any) -> Iterable[ComponentSpec]:
    loaded = value() if callable(value) else value
    if isinstance(loaded, ComponentSpec):
        return [loaded]
    return loaded


def discover_entry_points(
    registry: ComponentRegistry,
    group: str = "rlflow.components",
) -> None:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=group)
    else:
        selected = entry_points.get(group, [])

    for entry_point in selected:
        for component in _as_components(entry_point.load()):
            if registry.maybe_get(component.id) is None:
                registry.register(component)
