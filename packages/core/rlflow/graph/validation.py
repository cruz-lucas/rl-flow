from __future__ import annotations

from collections import defaultdict
from typing import Any

from jsonschema import Draft202012Validator

from rlflow.registry.base import ComponentRegistry
from rlflow.schemas.component import ComponentSpec
from rlflow.schemas.workflow import (
    ValidationErrorDetail,
    ValidationResult,
    WorkflowEdge,
    WorkflowSpec,
)


class WorkflowValidator:
    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry

    def validate(self, workflow: WorkflowSpec) -> ValidationResult:
        errors: list[ValidationErrorDetail] = []
        node_components: dict[str, ComponentSpec] = {}
        node_ids = set()

        for node in workflow.nodes:
            if node.id in node_ids:
                errors.append(
                    ValidationErrorDetail(
                        message=f"Duplicate node id: {node.id}",
                        node_id=node.id,
                        code="duplicate_node",
                    )
                )
                continue
            node_ids.add(node.id)
            component = self.registry.maybe_get(node.component)
            if component is None:
                errors.append(
                    ValidationErrorDetail(
                        message=f"Node references unknown component: {node.component}",
                        node_id=node.id,
                        field="component",
                        code="unknown_component",
                    )
                )
                continue
            node_components[node.id] = component
            errors.extend(self._validate_node_config(node.id, component, node.config))

        inbound: dict[tuple[str, str], list[WorkflowEdge]] = defaultdict(list)
        outbound: dict[tuple[str, str], list[WorkflowEdge]] = defaultdict(list)
        for edge in workflow.edges:
            errors.extend(self._validate_edge(edge, node_components))
            inbound[(edge.to_node, edge.to_port)].append(edge)
            outbound[(edge.from_node, edge.from_port)].append(edge)

        for node in workflow.nodes:
            component = node_components.get(node.id)
            if component is None:
                continue
            for port in component.input_ports:
                if port.required and not inbound.get((node.id, port.name)):
                    errors.append(
                        ValidationErrorDetail(
                            message=f"Required input port is not connected: {port.name}",
                            node_id=node.id,
                            field=port.name,
                            code="missing_required_port",
                        )
                    )

        runners = [
            node for node in workflow.nodes if node_components.get(node.id, None)
            and node_components[node.id].kind == "runner"
        ]
        if len(runners) != 1:
            errors.append(
                ValidationErrorDetail(
                    message=f"Workflow must contain exactly one runner node, found {len(runners)}",
                    code="runner_count",
                )
            )
        elif not self._runner_has_agent_and_environment(runners[0].id, workflow, node_components):
            errors.append(
                ValidationErrorDetail(
                    message="Runner must be connected to at least one agent and one environment",
                    node_id=runners[0].id,
                    code="runner_missing_agent_or_environment",
                )
            )
        else:
            errors.extend(self._validate_runner_semantics(runners[0].id, workflow, node_components))

        if workflow.execution.backend not in {"local", "slurm"}:
            errors.append(
                ValidationErrorDetail(
                    message=f"Unsupported execution backend: {workflow.execution.backend}",
                    field="execution.backend",
                    code="unsupported_backend",
                )
            )

        return ValidationResult(valid=not errors, errors=errors)

    def _validate_edge(
        self,
        edge: WorkflowEdge,
        node_components: dict[str, ComponentSpec],
    ) -> list[ValidationErrorDetail]:
        errors: list[ValidationErrorDetail] = []
        source = node_components.get(edge.from_node)
        target = node_components.get(edge.to_node)
        if source is None:
            errors.append(
                ValidationErrorDetail(
                    message=f"Edge references unknown source node: {edge.from_node}",
                    node_id=edge.from_node,
                    code="unknown_edge_node",
                )
            )
            return errors
        if target is None:
            errors.append(
                ValidationErrorDetail(
                    message=f"Edge references unknown target node: {edge.to_node}",
                    node_id=edge.to_node,
                    code="unknown_edge_node",
                )
            )
            return errors

        source_port = source.output_port(edge.from_port)
        target_port = target.input_port(edge.to_port)
        if source_port is None:
            errors.append(
                ValidationErrorDetail(
                    message=f"Unknown output port: {edge.from_port}",
                    node_id=edge.from_node,
                    field=edge.from_port,
                    code="unknown_port",
                )
            )
        if target_port is None:
            errors.append(
                ValidationErrorDetail(
                    message=f"Unknown input port: {edge.to_port}",
                    node_id=edge.to_node,
                    field=edge.to_port,
                    code="unknown_port",
                )
            )
        if source_port and target_port and source_port.type != target_port.type:
            errors.append(
                ValidationErrorDetail(
                    message=(
                        f"Port type mismatch: {edge.from_node}.{edge.from_port} "
                        f"({source_port.type}) -> {edge.to_node}.{edge.to_port} ({target_port.type})"
                    ),
                    node_id=edge.to_node,
                    field=edge.to_port,
                    code="port_type_mismatch",
                )
            )
        return errors

    def _validate_node_config(
        self,
        node_id: str,
        component: ComponentSpec,
        config: dict[str, Any],
    ) -> list[ValidationErrorDetail]:
        schema = component.config_schema or {"type": "object"}
        validator = Draft202012Validator(schema)
        errors = []
        for error in sorted(validator.iter_errors(config), key=lambda item: list(item.path)):
            path = ".".join(str(part) for part in error.path) or None
            errors.append(
                ValidationErrorDetail(
                    message=error.message,
                    node_id=node_id,
                    field=path,
                    code="invalid_config",
                )
            )
        return errors

    def _runner_has_agent_and_environment(
        self,
        runner_id: str,
        workflow: WorkflowSpec,
        node_components: dict[str, ComponentSpec],
    ) -> bool:
        connected_kinds = set()
        for edge in workflow.edges:
            if edge.to_node == runner_id:
                component = node_components.get(edge.from_node)
                if component is not None:
                    connected_kinds.add(component.kind)
        return "agent" in connected_kinds and "environment" in connected_kinds

    def _validate_runner_semantics(
        self,
        runner_id: str,
        workflow: WorkflowSpec,
        node_components: dict[str, ComponentSpec],
    ) -> list[ValidationErrorDetail]:
        errors: list[ValidationErrorDetail] = []
        inbound = {
            edge.to_port: edge
            for edge in workflow.edges
            if edge.to_node == runner_id
        }
        agent_edge = inbound.get("agent")
        if agent_edge is None:
            return errors

        agent_component = node_components.get(agent_edge.from_node)
        dqn_agents = {"builtin.agent.dqn_jax", "builtin.agent.dqn_rmax_jax"}
        if agent_component is None or agent_component.id not in dqn_agents:
            return errors

        replay_edge = inbound.get("replay_buffer")
        if replay_edge is None:
            errors.append(
                ValidationErrorDetail(
                    message="builtin DQN agents require builtin.replay.uniform on replay_buffer",
                    node_id=runner_id,
                    field="replay_buffer",
                    code="missing_required_port",
                )
            )
            return errors

        replay_component = node_components.get(replay_edge.from_node)
        if replay_component is not None and replay_component.id != "builtin.replay.uniform":
            errors.append(
                ValidationErrorDetail(
                    message="builtin DQN agents require builtin.replay.uniform",
                    node_id=runner_id,
                    field="replay_buffer",
                    code="invalid_component_connection",
                )
            )
        if agent_component.id == "builtin.agent.dqn_rmax_jax":
            knownness_edges = [
                edge
                for edge in workflow.edges
                if edge.to_node == agent_edge.from_node and edge.to_port == "knownness_signal"
            ]
            if len(knownness_edges) > 1:
                errors.append(
                    ValidationErrorDetail(
                        message="builtin.agent.dqn_rmax_jax accepts at most one knownness_signal input",
                        node_id=agent_edge.from_node,
                        field="knownness_signal",
                        code="multiple_input_edges",
                    )
                )
            if not knownness_edges and "intrinsic_reward" not in inbound:
                errors.append(
                    ValidationErrorDetail(
                        message=(
                            "builtin.agent.dqn_rmax_jax requires a knownness_signal "
                            "input, or a legacy runner intrinsic_reward input"
                        ),
                        node_id=agent_edge.from_node,
                        field="knownness_signal",
                        code="missing_required_port",
                    )
                )
        return errors
