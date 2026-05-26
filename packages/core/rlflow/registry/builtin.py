from __future__ import annotations

from rlflow.registry.base import ComponentRegistry
from rlflow.registry.discovery import discover_entry_points


def create_default_registry(discover: bool = True) -> ComponentRegistry:
    registry = ComponentRegistry()
    try:
        from rlflow_builtin.components import components as builtin_components

        registry.register_many(builtin_components())
    except ImportError:
        pass

    if discover:
        discover_entry_points(registry)
    return registry
