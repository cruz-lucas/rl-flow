from __future__ import annotations

from rlflow.schemas.component import ComponentSpec, PortSpec
from rlflow_builtin.component_schema import component_schema


def policy_components() -> list[ComponentSpec]:
    output = [PortSpec(name="policy", type="policy")]
    return [
        ComponentSpec(
            id="builtin.policy.epsilon_greedy",
            source="builtin",
            kind="policy",
            display_name="Epsilon-Greedy",
            description="Selects a random action with epsilon probability, otherwise greedy.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "epsilon": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "eval_epsilon": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                }
            ),
            defaults={"epsilon": 0.1, "eval_epsilon": 0.0},
        ),
        ComponentSpec(
            id="builtin.policy.ucb",
            source="builtin",
            kind="policy",
            display_name="UCB",
            description="Upper Confidence Bound exploration over action values.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "coefficient": {"type": "number", "minimum": 0.0},
                    "initial_count": {"type": "number", "exclusiveMinimum": 0.0},
                }
            ),
            defaults={"coefficient": 1.0, "initial_count": 1.0},
        ),
        ComponentSpec(
            id="builtin.policy.softmax",
            source="builtin",
            kind="policy",
            display_name="Softmax",
            description="Boltzmann exploration over action values.",
            output_ports=output,
            config_schema=component_schema(
                {
                    "temperature": {"type": "number", "exclusiveMinimum": 0.0},
                    "eval_temperature": {"type": "number", "exclusiveMinimum": 0.0},
                }
            ),
            defaults={"temperature": 1.0, "eval_temperature": 0.01},
        ),
    ]
