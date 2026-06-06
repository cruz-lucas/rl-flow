# API Endpoints

The FastAPI app is created in `rlflow_api.main:create_app`. It exposes the core compiler, registry, storage, executor, artifact, dataset, environment-session, and sweep functionality over HTTP.

## Health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check. |

## Components

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/components` | List registered component specs. |

## Workflows

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/workflows` | List saved workflows. |
| `POST` | `/workflows` | Save a workflow. |
| `GET` | `/workflows/{workflow_id}` | Load a saved workflow. |
| `GET` | `/workflows/examples/{name}` | Load an example workflow from `configs/workflows`. |
| `POST` | `/workflows/validate` | Validate a workflow. |
| `POST` | `/workflows/compile` | Compile a workflow into an experiment. |

## Experiments and Jobs

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/experiments/run` | Compile and submit a workflow. |
| `GET` | `/experiments/{experiment_id}` | Fetch experiment metadata. |
| `GET` | `/jobs` | List jobs. |
| `GET` | `/jobs/{job_id}` | Fetch one job. |
| `POST` | `/jobs/{job_id}/cancel` | Request cancellation. |

## Artifacts

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/artifacts/{experiment_id}` | List artifact paths for an experiment. |

## Environment Sessions

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/environment-sessions` | Create an interactive environment session. |
| `POST` | `/environment-sessions/{session_id}/actions` | Step the environment. |
| `POST` | `/environment-sessions/{session_id}/reset` | Reset the environment. |
| `GET` | `/environment-sessions/{session_id}/export.pdf` | Export a symbolic grid snapshot. |

## Datasets

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/datasets` | List `.npz` datasets under the run root. |
| `POST` | `/datasets/inspect` | Inspect arrays and transition visitation. |
| `POST` | `/datasets/offline-rnd` | Run offline novelty analysis. |

## Sweeps

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/sweeps` | List compiled sweep manifests. |
| `POST` | `/sweeps/inspect` | Summarize a compiled sweep. |
| `GET` | `/sweeps/workflows/{workflow_id}/candidates` | Suggest sweepable workflow fields. |
| `POST` | `/sweeps/compile` | Compile a sweep from a saved workflow. |
| `POST` | `/sweeps/run` | Compile and submit a sweep. |

Use FastAPI's generated OpenAPI route while the server is running for exact request and response schemas:

```text
http://127.0.0.1:8000/docs
```
