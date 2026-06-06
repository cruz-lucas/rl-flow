# Quickstart

## Install

```bash
uv sync --extra dev --extra docs
pnpm install
```

Use the `analysis` extra if you only need plotting and reports without the docs toolchain:

```bash
uv sync --extra analysis
```

## Run a Smoke Experiment

RiverSwim Q-learning is the fastest end-to-end check.

```bash
uv run rlflow workflow validate configs/workflows/tabular_q_learning_riverswim.yaml
uv run rlflow run configs/workflows/tabular_q_learning_riverswim.yaml --backend local
```

The command prints a job record and writes a run under `runs/`. Inspect the run directory for `workflow.yaml`, `resolved_config.yaml`, `generated.gin`, `command.sh`, `manifest.json`, logs, summaries, and optional checkpoints.

## Start the API

```bash
uv run uvicorn rlflow_api.main:app --reload
```

Useful endpoints include:

- `GET /health`
- `GET /components`
- `POST /workflows/validate`
- `POST /workflows/compile`
- `POST /experiments/run`
- `GET /jobs/{job_id}`
- `GET /sweeps`
- `POST /datasets/inspect`

## Start the UI

```bash
pnpm --filter @rl-flow/web dev
```

Open `http://localhost:5173`. The UI includes workflow editing, job views, environment inspection, dataset inspection, offline intrinsic-reward analysis, and sweep construction.

## Build the Docs

```bash
uv run python scripts/generate_docs_reference.py
uv run mkdocs build --strict
```

Use `uv run mkdocs serve` for local browsing.
