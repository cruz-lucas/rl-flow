from __future__ import annotations

from collections.abc import Iterable

from rlflow.schemas.component import ComponentKind, ComponentSpec


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, ComponentSpec] = {}

    def register(self, component: ComponentSpec) -> None:
        if component.id in self._components:
            raise ValueError(f"Component already registered: {component.id}")
        self._components[component.id] = component

    def register_many(self, components: Iterable[ComponentSpec]) -> None:
        for component in components:
            self.register(component)

    def get(self, component_id: str) -> ComponentSpec:
        try:
            return self._components[component_id]
        except KeyError as exc:
            raise KeyError(f"Unknown component: {component_id}") from exc

    def maybe_get(self, component_id: str) -> ComponentSpec | None:
        return self._components.get(component_id)

    def list(self) -> list[ComponentSpec]:
        return [self._components[key] for key in sorted(self._components)]

    def list_by_kind(self, kind: ComponentKind) -> list[ComponentSpec]:
        return [component for component in self.list() if component.kind == kind]

    def list_by_source(self, source: str) -> list[ComponentSpec]:
        return [component for component in self.list() if component.source == source]

    def list_by_source_and_kind(self, source: str, kind: ComponentKind) -> list[ComponentSpec]:
        return [
            component
            for component in self.list()
            if component.source == source and component.kind == kind
        ]
