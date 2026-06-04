from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

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
from rlflow.tracking.status import RunStatusState, load_status

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


def _cli_error_message(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError) and exc.name in {"pandas", "matplotlib", "tabulate"}:
        return (
            f"{exc.name} is required for analysis commands. Install analysis dependencies "
            "with `uv sync --extra analysis` or run with `uv run --extra analysis ...`."
        )
    return str(exc)


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
    metric_last_n: int | None = typer.Option(None, "--metric-last-n", min=1),
) -> None:
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    if goal is not None and goal not in {"maximize", "minimize"}:
        typer.echo("--goal must be maximize or minimize", err=True)
        raise typer.Exit(code=1)
    summary = SweepCompiler(_registry()).summarize(path, metric=metric, goal=goal, metric_last_n=metric_last_n)
    typer.echo(yaml.safe_dump(summary, sort_keys=True))


@sweep_app.command("status")
def sweep_status(path: Path) -> None:
    manifest_path = path / "sweep_manifest.yaml" if path.is_dir() else path
    try:
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        compilation = SweepCompiler(_registry()).summarize(manifest_path)
    except Exception as exc:
        typer.echo(_cli_error_message(exc), err=True)
        raise typer.Exit(code=1) from exc

    from rlflow.schemas.sweep import SweepCompilation

    sweep_compilation = SweepCompilation.model_validate(manifest_data)
    counts: Counter[str] = Counter()
    failed_details: list[str] = []
    for trial in sweep_compilation.trials:
        state = _trial_filesystem_state(Path(trial.run_dir))
        counts[state] += 1
        if state == RunStatusState.failed.value:
            hint = _trial_failure_hint(Path(trial.run_dir))
            failed_details.append(
                f"{trial.trial_id}: {hint}" if hint else trial.trial_id
            )

    best = compilation.get("best") if isinstance(compilation, dict) else None
    best_group = best.get("group_id") if isinstance(best, dict) else None
    typer.echo(f"sweep: {sweep_compilation.sweep_id}")
    typer.echo(f"total trials: {len(sweep_compilation.trials)}")
    for state in [
        RunStatusState.completed.value,
        RunStatusState.failed.value,
        RunStatusState.running.value,
        RunStatusState.queued.value,
        RunStatusState.compiled.value,
        RunStatusState.cancelled.value,
        RunStatusState.unknown.value,
        "missing",
    ]:
        typer.echo(f"{state}: {counts[state]}")
    typer.echo(f"best completed group: {best_group or '-'}")
    if failed_details:
        typer.echo("failed details:")
        for detail in failed_details:
            typer.echo(f"- {detail}")


@sweep_app.command("report")
def sweep_report(
    path: Path,
    metric: str | None = typer.Option(None, "--metric"),
    goal: str | None = typer.Option(None, "--goal", help="maximize or minimize"),
    metric_last_n: int | None = typer.Option(None, "--metric-last-n", min=1),
    top_k: int = typer.Option(20, "--top-k", min=1, help="Number of ranked groups to print."),
    all_groups: bool = typer.Option(False, "--all", help="Print all ranked groups."),
    show_trials: bool = typer.Option(False, "--show-trials", help="Include per-trial metric rows."),
    out: Path | None = typer.Option(None, "--out", help="Optional directory for report/json/csv files."),
) -> None:
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    if goal is not None and goal not in {"maximize", "minimize"}:
        typer.echo("--goal must be maximize or minimize", err=True)
        raise typer.Exit(code=1)

    try:
        from rlflow.analysis.report import export_sweep_report, format_sweep_report

        resolved_top_k = None if all_groups else top_k
        summary = SweepCompiler(_registry()).summarize(path, metric=metric, goal=goal, metric_last_n=metric_last_n)
        typer.echo(format_sweep_report(summary, top_k=resolved_top_k, include_trials=show_trials))

        if out is not None:
            paths = export_sweep_report(
                summary,
                out_dir=out,
                top_k=resolved_top_k,
                include_trials=show_trials,
            )
            typer.echo("")
            typer.echo("Wrote files:")
            for label, output_path in paths.items():
                typer.echo(f"{label}: {output_path}")
    except Exception as exc:
        typer.echo(_cli_error_message(exc), err=True)
        raise typer.Exit(code=1) from exc


@sweep_app.command("export-learning-curves")
def sweep_export_learning_curves(
    path: Path,
    out: Path | None = typer.Option(None, "--out"),
    history: str = typer.Option("train", "--history", help="train or eval"),
    value: str = typer.Option("discounted_return", "--value"),
    bootstrap_samples: int = typer.Option(1000, "--bootstrap-samples", min=0),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    if path.is_dir():
        path = path / "sweep_manifest.yaml"
    try:
        result = SweepCompiler(_registry()).export_learning_curves(
            path,
            out_dir=out,
            history=history,
            value=value,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except SweepCompilationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(yaml.safe_dump(result, sort_keys=True))


@sweep_app.command("plot-learning-curves", context_settings={"allow_extra_args": True})
def sweep_plot_learning_curves(
    ctx: typer.Context,
    path: Path,
    out: Path | None = typer.Option(None, "--out"),
    config: Path | None = typer.Option(None, "--config"),
    history: str | None = typer.Option(None, "--history"),
    x: str | None = typer.Option(None, "--x"),
    y: str | None = typer.Option(None, "--y", "--value"),
    points: int | None = typer.Option(None, "--points", min=2),
    bootstrap_samples: int | None = typer.Option(None, "--bootstrap-samples", min=0),
    seed: int | None = typer.Option(None, "--seed"),
    top_k: int | None = typer.Option(None, "--top-k", min=1),
    sort_by: str | None = typer.Option(None, "--sort-by"),
    goal: str | None = typer.Option(None, "--goal"),
    groups: list[str] | None = typer.Option(None, "--groups"),
    smooth_window: int | None = typer.Option(None, "--smooth-window", min=1),
) -> None:
    try:
        from rlflow.analysis.curves import (
            apply_curve_labels,
            interpolate_and_aggregate,
            smooth_curve_columns,
        )
        from rlflow.analysis.loading import load_histories, load_sweep_manifest
        from rlflow.analysis.plotting import plot_learning_curves
        from rlflow.analysis.summary import summarize_groups

        config_data = _load_plot_config(config)
        curve_config = _resolve_curve_config(
            config_data.get("curves", {}),
            {
                "history": history,
                "x": x,
                "y": y,
                "points": points,
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
                "top_k": top_k,
                "sort_by": sort_by,
                "goal": goal,
                "smooth_window": smooth_window,
            },
        )
        cli_groups = _resolve_cli_groups(groups, ctx.args)
        configured_groups = _as_string_list(curve_config.get("groups"))
        selected_groups = cli_groups if cli_groups is not None else configured_groups
        curve_config["groups"] = selected_groups

        if selected_groups and curve_config["top_k"] is not None:
            raise ValueError("--groups and --top-k are mutually exclusive")
        if curve_config["top_k"] is not None and curve_config["sort_by"] is None:
            raise ValueError("--top-k requires --sort-by")
        if curve_config["goal"] not in {"maximize", "minimize"}:
            raise ValueError("--goal must be maximize or minimize")

        compilation = load_sweep_manifest(path)
        out_dir = out or Path(compilation.sweep_dir) / "analysis" / "plots"

        raw = load_histories(path, history=curve_config["history"])
        if raw.empty:
            raise ValueError("No histories found for this sweep")

        curves = interpolate_and_aggregate(
            raw,
            x=curve_config["x"],
            y=curve_config["y"],
            points=curve_config["points"],
            bootstrap_samples=curve_config["bootstrap_samples"],
            seed=curve_config["seed"],
        )
        if curves.empty:
            raise ValueError("No curve data to export")

        if selected_groups is None and curve_config["top_k"] is not None:
            summary = summarize_groups(
                path,
                metric=curve_config["sort_by"],
                goal=curve_config["goal"],
            )
            if summary.empty:
                raise ValueError(f"No completed group found for metric: {curve_config['sort_by']}")
            selected_groups = list(summary.head(curve_config["top_k"])["group_id"])
            curve_config["groups"] = selected_groups

        if selected_groups is not None:
            keep = set(selected_groups)
            curves = curves[curves["group_id"].isin(keep)].copy()
            raw = _filter_raw_histories(raw, keep)
            if curves.empty:
                raise ValueError("Group filtering removed all curve data")

        labels = config_data.get("labels", {}) or {}
        if not isinstance(labels, dict):
            raise ValueError("plot config labels must be a mapping")
        curves = apply_curve_labels(curves, labels)
        plot_curves = smooth_curve_columns(curves, window=curve_config["smooth_window"])

        figure_config = _resolve_figure_config(
            config_data.get("figure", {}),
            x=curve_config["x"],
            y=curve_config["y"],
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = out_dir / "curves_raw.csv"
        interpolated_csv = out_dir / "curves_interpolated.csv"
        curves_csv = out_dir / "curves.csv"
        config_path = out_dir / "plot_config.yaml"
        _write_dataframe_csv(raw, raw_csv)
        _write_dataframe_csv(curves, interpolated_csv)
        _write_dataframe_csv(curves, curves_csv)

        plot_curves_csv = None
        if curve_config["smooth_window"] and curve_config["smooth_window"] > 1:
            plot_curves_csv = out_dir / "curves_plot.csv"
            _write_dataframe_csv(plot_curves, plot_curves_csv)

        paths = plot_learning_curves(
            plot_curves,
            out_dir=out_dir,
            x_label=figure_config["x_label"],
            y_label=figure_config["y_label"],
            title=figure_config["title"],
            legend_title=figure_config["legend_title"],
            width=figure_config["width"],
            height=figure_config["height"],
            dpi=figure_config["dpi"],
        )
        resolved_config = {
            "figure": figure_config,
            "curves": curve_config,
            "labels": labels,
        }
        config_path.write_text(
            yaml.safe_dump(resolved_config, sort_keys=False),
            encoding="utf-8",
        )

        result = {
            "curves_raw_csv": str(raw_csv),
            "curves_interpolated_csv": str(interpolated_csv),
            "curves_csv": str(curves_csv),
            "plot_config": str(config_path),
            "plots": {key: str(value) for key, value in paths.items()},
        }
        if plot_curves_csv is not None:
            result["curves_plot_csv"] = str(plot_curves_csv)
        typer.echo(yaml.safe_dump(result, sort_keys=True))
    except Exception as exc:
        typer.echo(_cli_error_message(exc), err=True)
        raise typer.Exit(code=1) from exc


@sweep_app.command("export-summary")
def sweep_export_summary(
    path: Path,
    metric: str = typer.Option(..., "--metric"),
    goal: str = typer.Option("maximize", "--goal"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    try:
        from rlflow.analysis.loading import load_sweep_manifest
        from rlflow.analysis.summary import export_summary_tables, summarize_groups

        compilation = load_sweep_manifest(path)
        out_dir = out or Path(compilation.sweep_dir) / "analysis"
        summary = summarize_groups(path, metric=metric, goal=goal)
        if summary.empty:
            raise ValueError(f"No completed group found for metric: {metric}")
        paths = export_summary_tables(summary, out_dir=out_dir)
        typer.echo(yaml.safe_dump(paths, sort_keys=True))
    except Exception as exc:
        typer.echo(_cli_error_message(exc), err=True)
        raise typer.Exit(code=1) from exc


@sweep_app.command("export-best")
def sweep_export_best(
    path: Path,
    metric: str = typer.Option(..., "--metric"),
    goal: str = typer.Option("maximize", "--goal"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    try:
        from rlflow.analysis.best import export_best_config

        paths = export_best_config(path, metric=metric, goal=goal, out_dir=out)
        typer.echo(yaml.safe_dump(paths, sort_keys=True))
    except Exception as exc:
        typer.echo(_cli_error_message(exc), err=True)
        raise typer.Exit(code=1) from exc


def _trial_filesystem_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "missing"
    status = load_status(run_dir)
    if status is not None:
        return status.status.value
    if (
        (run_dir / "summaries" / "metrics.json").exists()
        or (run_dir / "metrics.json").exists()
        or (run_dir / "logs" / "train_history.jsonl").exists()
        or (run_dir / "logs" / "eval_history.jsonl").exists()
    ):
        return RunStatusState.completed.value
    return RunStatusState.compiled.value


def _trial_failure_hint(run_dir: Path) -> str | None:
    candidates = [
        run_dir / "logs" / "stderr.log",
        run_dir / "logs" / "local.err",
        *(sorted((run_dir / "logs").glob("slurm-*.err")) if (run_dir / "logs").exists() else []),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        except OSError:
            continue
        lines = [line for line in lines if line]
        if lines:
            return lines[-1]
    return None


def _load_plot_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("plot config must be a YAML mapping")
    return data


def _resolve_curve_config(
    config_section: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    if config_section is None:
        config_section = {}
    if not isinstance(config_section, dict):
        raise ValueError("plot config curves section must be a mapping")

    config = {
        "history": "train",
        "x": "env_step",
        "y": "discounted_return",
        "points": 500,
        "bootstrap_samples": 1000,
        "seed": 0,
        "top_k": None,
        "sort_by": None,
        "goal": "maximize",
        "groups": None,
        "smooth_window": None,
    }
    if "value" in config_section and "y" not in config_section:
        config_section = {**config_section, "y": config_section["value"]}
    for key, value in config_section.items():
        if key in config:
            config[key] = value
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["points"] = int(config["points"])
    config["bootstrap_samples"] = int(config["bootstrap_samples"])
    config["seed"] = int(config["seed"])
    config["top_k"] = None if config["top_k"] is None else int(config["top_k"])
    config["smooth_window"] = (
        None if config["smooth_window"] is None else int(config["smooth_window"])
    )
    config["history"] = str(config["history"])
    config["x"] = str(config["x"])
    config["y"] = str(config["y"])
    config["goal"] = str(config["goal"])
    return config


def _resolve_figure_config(config_section: Any, *, x: str, y: str) -> dict[str, Any]:
    if config_section is None:
        config_section = {}
    if not isinstance(config_section, dict):
        raise ValueError("plot config figure section must be a mapping")

    config = {
        "width": 3.25,
        "height": 2.35,
        "dpi": 300,
        "title": None,
        "x_label": _axis_label(x),
        "y_label": _axis_label(y),
        "legend_title": None,
    }
    for key, value in config_section.items():
        if key in config:
            config[key] = value
    config["width"] = float(config["width"])
    config["height"] = float(config["height"])
    config["dpi"] = int(config["dpi"])
    return config


def _axis_label(name: str) -> str:
    labels = {
        "env_step": "Environment steps",
        "episode": "Episode",
        "return": "Return",
        "discounted_return": "Discounted return",
        "loss": "Loss",
        "length": "Episode length",
    }
    return labels.get(name, name.replace("_", " ").title())


def _resolve_cli_groups(
    groups: list[str] | None,
    extra_args: list[str],
) -> list[str] | None:
    selected = list(groups or [])
    if extra_args:
        if not selected:
            raise ValueError(f"Unexpected extra arguments: {' '.join(extra_args)}")
        selected.extend(extra_args)
    return selected or None


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError("groups must be a string or list of strings")


def _filter_raw_histories(raw: Any, keep: set[str]) -> Any:
    if "group_id" not in raw.columns:
        return raw
    present = raw["group_id"].dropna()
    if present.empty:
        return raw
    return raw[raw["group_id"].astype(str).isin(keep)].copy()


def _write_dataframe_csv(df: Any, path: Path) -> None:
    export_df = df.copy()
    for column in export_df.columns:
        export_df[column] = export_df[column].apply(_csv_cell)
    export_df.to_csv(path, index=False)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


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
