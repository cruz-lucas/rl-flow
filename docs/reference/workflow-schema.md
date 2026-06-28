# Workflow Schema

Workflow schemas are Pydantic models in `rlflow.schemas.workflow`.

## `WorkflowSpec`

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Human-readable workflow name. |
| `description` | string | Optional description. |
| `nodes` | list of `WorkflowNode` | Component instances. |
| `edges` | list of `WorkflowEdge` | Typed graph connections. |
| `execution` | `ExecutionSpec` | Local or SLURM backend settings. |
| `metadata` | object | Experiment, sweep, and seed metadata. |

## `WorkflowNode`

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable node ID used by edges and sweep targets. |
| `component` | string | Component ID in the registry. |
| `config` | object | Overrides merged with component defaults. |
| `position` | object | UI coordinates. |

## `WorkflowEdge`

| Field | Type | Description |
| --- | --- | --- |
| `from_node` | string | Source node ID. |
| `from_port` | string | Source output port. |
| `to_node` | string | Target node ID. |
| `to_port` | string | Target input port. |

## `ExecutionSpec`

| Field | Type | Description |
| --- | --- | --- |
| `backend` | `local` or `slurm` | Execution backend. |
| `cluster` | string or null | Optional cluster name. |
| `options` | object | Backend-specific options. |

## Validation Rules

Validation rejects:

- duplicate node IDs
- unknown components
- invalid node config according to JSON schema
- unknown edge nodes
- unknown ports
- port type mismatches
- missing required input ports
- workflows without exactly one runner
- runners without an agent and environment
- DQN workflows without the required uniform replay buffer
- DQN + R-Max workflows without a knownness signal
- unsupported execution backends
