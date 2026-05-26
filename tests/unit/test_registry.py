import pytest

from rlflow.registry.base import ComponentRegistry
from rlflow.schemas.component import ComponentSpec


def spec(component_id: str = "test.agent") -> ComponentSpec:
    return ComponentSpec(
        id=component_id,
        kind="agent",
        display_name="Test Agent",
        output_ports=[],
    )


def test_registry_register_get_and_filter() -> None:
    registry = ComponentRegistry()
    registry.register(spec())

    assert registry.get("test.agent").display_name == "Test Agent"
    assert [item.id for item in registry.list_by_kind("agent")] == ["test.agent"]
    assert [item.id for item in registry.list_by_source("custom")] == ["test.agent"]
    assert [item.id for item in registry.list_by_source_and_kind("custom", "agent")] == ["test.agent"]


def test_registry_rejects_duplicate_ids() -> None:
    registry = ComponentRegistry()
    registry.register(spec())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec())
