from __future__ import annotations

import subprocess
from pathlib import Path

import typer
import yaml

from rlflow.execution.local import LocalExecutor
from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.compiler import WorkflowCompilationError, WorkflowCompiler
from rlflow.graph.run_naming import make_flow_run_dir, make_run_id
from rlflow.graph.sweep import SweepCompilationError, SweepCompiler
from rlflow.graph.validation import WorkflowValidator
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.sweep import SweepSpec
from rlflow.schemas.workflow import ExecutionSpec, WorkflowSpec
from rlflow.storage.sqlite import Storage

app = typer.Typer(no_args_is_help=True)
components_app = typer.Typer(no_args_is_help=True)
workflow_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
sweep_app = typer.Typer(no_args_is_help=True)
app.add_typer(components_app, name="components")
app.add_typer(workflow_app, name="workflow")
app.add_typer(jobs_app, name="jobs")
app.add_typer(sweep_app, name="sweep")


def _registry():
    return create_default_registry()


def _load_workflow(path: Path) -> WorkflowSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return WorkflowSpec.model_validate(data)


def _load_sweep(path: Path) -> SweepSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SweepSpec.model_validate(data)


def _storage() -> Storage:
    storage = Storage.from_path("runs/rlflow.db")
    storage.init()
    return storage


@components_app.command("list")
def list_components(
    kind: str | None = typer.Option(None, help="Optional component kind filter."),
    source: str | None = typer.Option(None, help="Optional component source filter."),
) -> None:
    registry = _registry()
    if source and kind:
        components = registry.list_by_source_and_kind(source, kind)
    elif source:
        components = registry.list_by_source(source)
    elif kind:
        components = registry.list_by_kind(kind)
    else:
        components = registry.list()
    for component in components:
        typer.echo(f"{component.source}\t{component.id}\t{component.kind}\t{component.display_name}")


@workflow_app.command("validate")
def workflow_validate(path: Path) -> None:
    result = WorkflowValidator(_registry()).validate(_load_workflow(path))
    if result.valid:
        typer.echo("Workflow is valid")
        return
    for error in result.errors:
        node = f" node={error.node_id}" if error.node_id else ""
        field = f" field={error.field}" if error.field else ""
        typer.echo(f"{error.code}:{node}{field} {error.message}", err=True)
    raise typer.Exit(code=1)


@workflow_app.command("compile")
def workflow_compile(path: Path, out: Path = typer.Option(..., "--out")) -> None:
    _compile(path, out)


@app.command("compile")
def compile_shortcut(path: Path, out: Path = typer.Option(..., "--out")) -> None:
    _compile(path, out)


def _compile(path: Path, out: Path) -> None:
    workflow = _load_workflow(path)
    try:
        experiment = WorkflowCompiler(_registry()).compile(workflow, out_dir=out)
    except WorkflowCompilationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _storage().save_experiment(experiment)
    typer.echo(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=True))


@app.command("run")
def run_shortcut(
    path: Path,
    backend: str = typer.Option("local", "--backend", help="local or slurm"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    run_workflow(path, backend=backend, out=out)


@workflow_app.command("run")
def run_workflow(
    path: Path,
    backend: str = typer.Option("local", "--backend", help="local or slurm"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    workflow = _load_workflow(path)
    workflow.execution.backend = backend
    if out is None:
        run_id = make_run_id(workflow.name)
        workflow.metadata = {**workflow.metadata, "experiment_id": run_id}
        out = make_flow_run_dir("runs", workflow.name, run_id)
    compiler = WorkflowCompiler(_registry())
    experiment = compiler.compile(workflow, out_dir=out)
    storage = _storage()
    storage.save_experiment(experiment)
    executor = LocalExecutor() if backend == "local" else SlurmExecutor()
    job = executor.submit(experiment)
    storage.save_job(job)
    typer.echo(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=True))


@jobs_app.command("list")
def jobs_list() -> None:
    for job in _storage().list_jobs():
        typer.echo(f"{job.job_id}\t{job.status}\t{job.experiment_id}\t{job.run_dir}")


@jobs_app.command("status")
def jobs_status(job_id: str) -> None:
    record = _storage().get_job(job_id)
    if record is None:
        typer.echo(f"Unknown job: {job_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{record.job_id}\t{record.status}\t{record.message}")


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    record = _storage().get_job(job_id)
    if record is None:
        typer.echo(f"Unknown job: {job_id}", err=True)
        raise typer.Exit(code=1)
    executor = LocalExecutor() if record.backend == "local" else SlurmExecutor()
    executor.cancel(job_id)
    typer.echo(f"Cancel requested for {job_id}")


@sweep_app.command("compile")
def sweep_compile(
    path: Path,
    out: Path | None = typer.Option(None, "--out"),
    backend: str | None = typer.Option(None, "--backend", help="Optional backend override: local or slurm"),
) -> None:
    spec = _load_sweep(path)
    _override_sweep_backend(spec, backend)
    try:
        compilation = SweepCompiler(_registry()).compile(spec, base_path=path.parent, out_dir=out)
    except SweepCompilationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(yaml.safe_dump(compilation.model_dump(mode="json"), sort_keys=True))


@sweep_app.command("run")
def sweep_run(
    path: Path,
    out: Path | None = typer.Option(None, "--out"),
    backend: str | None = typer.Option(None, "--backend", help="Optional backend override: local or slurm"),
) -> None:
    spec = _load_sweep(path)
    _override_sweep_backend(spec, backend)
    try:
        compilation = SweepCompiler(_registry()).compile(spec, base_path=path.parent, out_dir=out)
    except SweepCompilationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    compiled_backend = _compiled_sweep_backend(compilation.trials[0].workflow_path)
    if compiled_backend == "slurm":
        if compilation.slurm_array_path is None:
            typer.echo("Sweep compiled for SLURM but no slurm_array.sh was generated", err=True)
            raise typer.Exit(code=1)
        try:
            job = SlurmExecutor().submit_array(compilation)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _storage().save_job(job)
        typer.echo(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=True))
        return

    if compiled_backend == "local":
        for trial in compilation.trials:
            try:
                subprocess.run(
                    ["/usr/bin/env", "bash", trial.command],
                    cwd=trial.run_dir,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                typer.echo(f"Trial {trial.trial_id} failed with exit code {exc.returncode}", err=True)
                raise typer.Exit(code=exc.returncode) from exc
        typer.echo(yaml.safe_dump(compilation.model_dump(mode="json"), sort_keys=True))
        return

    typer.echo(f"Unsupported sweep backend: {compiled_backend}", err=True)
    raise typer.Exit(code=1)


@sweep_app.command("summarize")
def sweep_summarize(
    path: Path,
    metric: str | None = typer.Option(None, "--metric"),
    goal: str | None = typer.Option(None, "--goal", help="maximize or minimize"),
) -> None:
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    if goal is not None and goal not in {"maximize", "minimize"}:
        typer.echo("--goal must be maximize or minimize", err=True)
        raise typer.Exit(code=1)
    summary = SweepCompiler(_registry()).summarize(path, metric=metric, goal=goal)
    typer.echo(yaml.safe_dump(summary, sort_keys=True))


def _override_sweep_backend(spec: SweepSpec, backend: str | None) -> None:
    if backend is None:
        return
    if backend not in {"local", "slurm"}:
        typer.echo(f"Unsupported backend: {backend}", err=True)
        raise typer.Exit(code=1)
    if spec.execution is None:
        spec.execution = ExecutionSpec(backend=backend)
    else:
        spec.execution.backend = backend


def _compiled_sweep_backend(workflow_path: str) -> str:
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8"))
    )
    return workflow.execution.backend


if __name__ == "__main__":
    app()
