# Mental Model

`rl-flow` has four layers:

1. Component specs describe what can be connected.
2. Workflows connect component instances into an experiment graph.
3. Compilation turns that graph into deterministic run artifacts.
4. Execution and analysis consume those artifacts.

```mermaid
flowchart TB
    specs[Component specs] --> graph[Workflow graph]
    graph --> validation[Validation]
    validation --> compile[Compilation]
    compile --> files[Run files]
    files --> execute[Execution]
    execute --> histories[Histories and metrics]
    histories --> reports[Reports and plots]
```

## Component Specs

A component is a `ComponentSpec`. It declares:

- `kind`: agent, environment, runner, policy, replay buffer, intrinsic reward, logger, analysis, and related roles.
- input and output ports with explicit types.
- a JSON schema for configurable fields.
- defaults that are merged with node-level overrides.
- compile targets such as Gin bindings or a runner module.

The web UI renders forms from those schemas. Adding a new RL module should normally happen in Python, not React.

## Workflows

A workflow is a graph of nodes and edges. Each node points at a registered component and carries local config. Each edge connects a typed output port to a typed input port. Validation rejects unknown components, bad ports, incompatible port types, invalid config, unsupported backends, and invalid runner wiring.

## Compilation

Compilation creates a run directory and writes:

- `workflow.yaml`: original workflow specification.
- `resolved_config.yaml`: defaults merged with overrides.
- `generated.gin`: deterministic Gin bindings.
- `command.sh`: executable command for the selected runner.
- `manifest.json`: hashes, dependency versions, git state, backend, seed, and sweep metadata.
- `status.json`: current run lifecycle state.

## Execution

The local backend starts the generated command on the current machine. The SLURM backend renders `slurm_job.sh` or `slurm_array.sh` and submits through `sbatch` when available. Both paths rely on the same compiled run directory, which keeps local and HPC execution as close as possible.

## Analysis

Training and evaluation histories are written as JSONL. Sweep analysis reads trial manifests, groups seed replicates by non-seed hyperparameters, ranks groups by metric, and can export summaries and learning-curve plots.
