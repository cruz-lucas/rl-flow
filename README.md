# rl-flow

rl-flow is a modular GUI framework for reinforcement learning experiments. It lets users compose RL experiments as node workflows, then validates and compiles those workflows into reproducible run directories with Gin config, command scripts, and local or SLURM launch metadata.

The frontend is schema-driven: component forms come from backend JSON schemas, so adding an agent, environment, logger, buffer, or intrinsic reward module does not require editing React code.

## Install

```bash
uv sync --extra dev
```

## Start the API

```bash
uv run uvicorn rlflow_api.main:app --reload
```

The API exposes:

- `GET /health`
- `GET /components`
- `POST /workflows/validate`
- `POST /workflows/compile`
- `POST /experiments/run`
- `GET /jobs/{job_id}`

## Start the UI

```bash
pnpm install
pnpm --filter @rl-flow/web dev
```

Open `http://localhost:5173`.

## Validate and Compile a Navix DQN Workflow

```bash
uv run python -m rlflow.cli workflow validate configs/workflows/navix_dqn_empty_room.yaml
uv run python -m rlflow.cli compile configs/workflows/navix_dqn_empty_room.yaml --out runs/test
```

Run locally:

```bash
uv run python -m rlflow.cli run configs/workflows/navix_dqn_empty_room.yaml --backend local
```

The local command writes logs, resolved config, metrics, and optional checkpoints into the run directory.

## SLURM

Set workflow execution to `slurm` and provide options like:

```yaml
execution:
  backend: slurm
  options:
    partition: gpu
    account: project-account
    time: "01:00:00"
    cpus_per_task: 4
    mem: 16G
    gres: gpu:1
```

Compilation writes `slurm_job.sh`. Submission uses `sbatch` if available.

## Hyperparameter Sweeps

Sweeps are YAML files that point at a workflow and override node config paths. A grid sweep can be compiled into per-trial run directories plus one SLURM array script:

```bash
uv run python -m rlflow.cli sweep compile configs/sweeps/navix_dqn_compute_canada.yaml --out runs/sweeps/navix-dqn
```

When a sweep includes a seed parameter, seed trials for the same non-seed configuration are nested under one group directory, for example `trials/group-0000/seed-0` and `trials/group-0000/seed-1`. Summaries and result plots use that grouping to average seed replicates for the same workflow configuration.

On a Compute Canada / Alliance login node, edit the sweep `account`, modules, and environment setup, then submit the array from that cluster checkout:

```bash
uv run python -m rlflow.cli sweep run configs/sweeps/navix_dqn_compute_canada.yaml --out runs/sweeps/navix-dqn
```

After jobs finish:

```bash
uv run python -m rlflow.cli sweep summarize runs/sweeps/navix-dqn
uv run python -m rlflow.cli sweep summarize runs/sweeps/navix-dqn --metric mean_train_return_last_n --metric-last-n 50
uv run python -m rlflow.cli sweep report runs/sweeps/navix-dqn --metric mean_train_return_last_n --metric-last-n 50 --out runs/sweeps/navix-dqn/analysis
```

`sweep report` is intended for headless cluster sessions: it prints a ranked terminal table, averages seed replicates by non-seed hyperparameters, and optionally writes `sweep_report.txt`, `sweep_summary.json`, and `sweep_groups.csv`.

## Add a Component

Create a package exposing `rlflow.components`:

```toml
[project.entry-points."rlflow.components"]
my_components = "my_package.components:components"
```

Return `ComponentSpec` objects with ports, JSON schema, defaults, and compile targets. The UI will render the schema automatically.

## Current Limitations

- Local job recovery after API restart is limited.
- MLflow integration is an architecture stub.
- The UI is intentionally minimal and focused on schema-driven workflow construction.
