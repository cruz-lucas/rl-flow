from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def agent_components() -> list[ComponentSpec]:
    output = [PortSpec(name="agent", type="agent")]
    common_properties = {
        "learning_rate": {"type": "number", "exclusiveMinimum": 0.0},
        "discount": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "initial_q": {"type": "number"},
    }
    return [
        ComponentSpec(
            id="builtin.agent.q_learning_tabular",
            source="builtin",
            kind="agent",
            display_name="Q-Learning Tabular",
            description="JAX tabular off-policy Q-learning agent.",
            output_ports=output,
            config_schema=component_schema(common_properties),
            defaults={"learning_rate": 0.2, "discount": 0.99, "initial_q": 0.0},
        ),
        ComponentSpec(
            id="builtin.agent.sarsa_tabular",
            source="builtin",
            kind="agent",
            display_name="Sarsa Tabular",
            description="JAX tabular on-policy Sarsa agent.",
            output_ports=output,
            config_schema=component_schema(common_properties),
            defaults={"learning_rate": 0.2, "discount": 0.99, "initial_q": 0.0},
        ),
    ]
