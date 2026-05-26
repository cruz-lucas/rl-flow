# Workflow Schema

Core models:

- `PortSpec`: `name`, `type`, `required`, `description`
- `ComponentSpec`: `id`, `source`, `version`, `kind`, ports, `config_schema`, `defaults`, `compile_target`
- `WorkflowNode`: `id`, `component`, `config`, UI `position`
- `WorkflowEdge`: source node/port and target node/port
- `WorkflowSpec`: `name`, `description`, `nodes`, `edges`, `execution`, `metadata`
- `ExperimentSpec`: resolved workflow plus run directory, command, generated files, and backend

Component kinds:

`agent`, `environment`, `runner`, `policy`, `replay_buffer`, `network`, `intrinsic_reward`, `logger`, `sweeper`, `launcher`, `analysis`

Validation rejects unknown components, missing required inputs, bad ports, incompatible port types, invalid config fields, missing runner requirements, and unsupported backends.
