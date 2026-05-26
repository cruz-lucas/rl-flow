from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import ExecutionSpec, WorkflowEdge, WorkflowNode, WorkflowSpec
from rlflow.graph.validation import WorkflowValidator


def valid_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="valid",
        execution=ExecutionSpec(backend="local"),
        nodes=[
            WorkflowNode(id="env", component="navix.env.grid"),
            WorkflowNode(id="agent", component="builtin.agent.dqn_jax"),
            WorkflowNode(id="replay", component="builtin.replay.uniform"),
            WorkflowNode(id="runner", component="builtin.runner.tabular_jax"),
        ],
        edges=[
            WorkflowEdge(from_node="env", from_port="environment", to_node="runner", to_port="environment"),
            WorkflowEdge(from_node="agent", from_port="agent", to_node="runner", to_port="agent"),
            WorkflowEdge(from_node="replay", from_port="replay_buffer", to_node="runner", to_port="replay_buffer"),
        ],
    )


def validator() -> WorkflowValidator:
    return WorkflowValidator(create_default_registry(discover=False))


def test_validation_accepts_minimal_valid_workflow() -> None:
    assert validator().validate(valid_workflow()).valid


def test_validation_catches_missing_component() -> None:
    workflow = valid_workflow()
    workflow.nodes[0].component = "missing"

    result = validator().validate(workflow)

    assert not result.valid
    assert any(error.code == "unknown_component" for error in result.errors)


def test_validation_catches_bad_ports() -> None:
    workflow = valid_workflow()
    workflow.edges[0].from_port = "bad"

    result = validator().validate(workflow)

    assert not result.valid
    assert any(error.code == "unknown_port" for error in result.errors)


def test_validation_catches_incompatible_ports() -> None:
    workflow = valid_workflow()
    workflow.edges[0].from_node = "agent"
    workflow.edges[0].from_port = "agent"

    result = validator().validate(workflow)

    assert not result.valid
    assert any(error.code == "port_type_mismatch" for error in result.errors)


def test_validation_catches_config_errors() -> None:
    workflow = valid_workflow()
    workflow.nodes[1].config = {"discount": 2.0}

    result = validator().validate(workflow)

    assert not result.valid
    assert any(error.code == "invalid_config" for error in result.errors)


def test_validation_requires_intrinsic_for_dqn_rmax() -> None:
    workflow = valid_workflow()
    workflow.nodes[1].component = "builtin.agent.dqn_rmax_jax"

    result = validator().validate(workflow)

    assert not result.valid
    assert any(error.field == "intrinsic_reward" for error in result.errors)
