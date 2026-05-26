# Architecture

rl-flow is a schema-driven workflow system for reinforcement learning experiments.

The frontend owns interaction state only: node placement, selected node, and JSON-schema form rendering. RL semantics live in Python component specs, validation, compilation, and execution services.

## Packages

- `packages/core/rlflow`: schemas, registry, graph validation, compiler, execution backends, storage, CLI.
- `packages/builtin_components/rlflow_builtin`: builtin agents, environments, policies, replay buffers, and runners.
- `apps/api/rlflow_api`: FastAPI app over the core package.
- `apps/web`: Vite React app using React Flow, TanStack Query, and Zustand.

## Flow

1. Component specs are loaded from built-ins and `rlflow.components` entry points.
2. The UI fetches specs and renders nodes plus JSON-schema config forms.
3. Validation checks components, ports, runner requirements, backend, and config schemas.
4. Compilation writes `workflow.yaml`, `resolved_config.yaml`, `generated.gin`, and `command.sh`.
5. Execution backends submit the compiled experiment locally or through SLURM.

The compiler is deterministic where order matters, especially Gin output.
