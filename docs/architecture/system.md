# System Architecture

`rl-flow` separates interaction, validation, compilation, execution, and analysis.

```mermaid
flowchart LR
    subgraph Browser
        ui[React Flow UI]
        forms[JSON-schema forms]
        ui --> forms
    end

    subgraph API
        fastapi[FastAPI routes]
        storage[SQLite storage]
        artifact[Filesystem artifact store]
    end

    subgraph Core
        registry[Component registry]
        validator[Workflow validator]
        compiler[Workflow compiler]
        sweeps[Sweep compiler]
        analysis[Analysis modules]
    end

    subgraph Execution
        local[Local executor]
        slurm[SLURM executor]
    end

    ui --> fastapi
    fastapi --> registry
    fastapi --> validator
    fastapi --> compiler
    fastapi --> sweeps
    fastapi --> storage
    fastapi --> artifact
    compiler --> local
    compiler --> slurm
    sweeps --> local
    sweeps --> slurm
    local --> analysis
    slurm --> analysis
```

## Packages

`packages/core/rlflow`
: schemas, registry, graph validation, compiler, sweep compiler, execution backends, storage, tracking, analysis, CLI.

`packages/builtin_components/rlflow_builtin`
: builtin environments, tabular algorithms, DQN/R-Max training code, intrinsic rewards, replay buffers, and runner modules.

`apps/api/rlflow_api`
: FastAPI service over the core package.

`apps/web`
: Vite React app with React Flow, TanStack Query, Zustand, and schema-rendered forms.

## Workflow Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant API
    participant Registry
    participant Compiler
    participant Executor
    participant RunDir

    User->>UI: compose workflow
    UI->>API: validate workflow
    API->>Registry: load component specs
    API->>API: validate ports and config
    UI->>API: compile or run
    API->>Compiler: compile WorkflowSpec
    Compiler->>RunDir: write reproducible files
    API->>Executor: submit experiment
    Executor->>RunDir: update status and logs
```

## Design Boundary

The frontend owns interaction state. The Python backend owns RL semantics. This boundary is the right default for a research framework because it lets algorithm authors add components without frontend work, while still giving users a visual composition layer.
