from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor
from rlflow.registry.builtin import create_default_registry
from rlflow.storage.filesystem import FilesystemArtifactStore
from rlflow_api.db import create_storage
from rlflow_api.routes import artifacts, components, datasets, environment_sessions, experiments, jobs, sweeps, workflows
from rlflow_api.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="rl-flow API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.registry = create_default_registry()
    app.state.storage = create_storage(settings)
    app.state.artifacts = FilesystemArtifactStore(settings.run_root)
    app.state.local_executor = LocalExecutor()
    app.state.slurm_executor = SlurmExecutor()
    environment_sessions.ensure_environment_session_store(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(components.router)
    app.include_router(workflows.router)
    app.include_router(experiments.router)
    app.include_router(jobs.router)
    app.include_router(artifacts.router)
    app.include_router(environment_sessions.router)
    app.include_router(datasets.router)
    app.include_router(sweeps.router)
    return app


app = create_app()
