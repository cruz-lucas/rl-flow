from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ComponentKind = Literal[
    "agent",
    "environment",
    "runner",
    "policy",
    "replay_buffer",
    "network",
    "intrinsic_reward",
    "logger",
    "sweeper",
    "launcher",
    "analysis",
]


class PortSpec(BaseModel):
    name: str
    type: str
    required: bool = True
    description: str = ""


class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str = "custom"
    version: str = "0.1.0"
    kind: ComponentKind
    display_name: str
    description: str = ""
    input_ports: list[PortSpec] = Field(default_factory=list)
    output_ports: list[PortSpec] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    compile_target: dict[str, Any] = Field(default_factory=dict)

    def input_port(self, name: str) -> PortSpec | None:
        return next((port for port in self.input_ports if port.name == name), None)

    def output_port(self, name: str) -> PortSpec | None:
        return next((port for port in self.output_ports if port.name == name), None)
