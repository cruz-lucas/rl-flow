from rlflow.registry.base import ComponentRegistry
from rlflow.registry.discovery import discover_entry_points
from rlflow.schemas.component import ComponentSpec


class FakeEntryPoint:
    def load(self):
        return lambda: [
            ComponentSpec(id="plugin.logger", kind="logger", display_name="Plugin Logger"),
        ]


class FakeEntryPoints:
    def select(self, group: str):
        assert group == "rlflow.components"
        return [FakeEntryPoint()]


def test_entry_point_discovery_can_be_mocked(monkeypatch) -> None:
    monkeypatch.setattr("rlflow.registry.discovery.metadata.entry_points", lambda: FakeEntryPoints())
    registry = ComponentRegistry()

    discover_entry_points(registry)

    assert registry.get("plugin.logger").kind == "logger"
