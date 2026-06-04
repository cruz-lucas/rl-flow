from __future__ import annotations

import shlex
import stat
from pathlib import Path
from typing import Any

import yaml

from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.gin_writer import GinWriter
from rlflow.graph.run_naming import make_flow_run_dir, make_run_id
from rlflow.graph.validation import WorkflowValidator
from rlflow.registry.base import ComponentRegistry
from rlflow.schemas.component import ComponentSpec
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.workflow import WorkflowSpec
from rlflow.tracking.manifest import (
    RunManifest,
    collect_dependency_versions,
    collect_git_info,
    file_sha256,
    platform_string,
    python_version,
    utc_now_iso,
    write_manifest,
)
from rlflow.tracking.status import RunStatusState, update_status


class WorkflowCompilationError(ValueError):
    pass


class WorkflowCompiler:
    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry
        self.validator = WorkflowValidator(registry)
        self.gin_writer = GinWriter()

    def compile(self, workflow: WorkflowSpec, out_dir: str | Path | None = None) -> ExperimentSpec:
        validation = self.validator.validate(workflow)
        if not validation.valid:
            messages = "; ".join(error.message for error in validation.errors)
            raise WorkflowCompilationError(f"Workflow is invalid: {messages}")

        experiment_id = str(workflow.metadata.get("experiment_id") or self._experiment_id(workflow.name))
        run_dir = Path(out_dir) if out_dir else make_flow_run_dir(
            "runs",
            workflow.name,
            experiment_id,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = run_dir.resolve()
        self._create_run_directories(run_dir)

        components = {node.id: self.registry.get(node.component) for node in workflow.nodes}
        resolved_config = {
            node.id: self._merge_dicts(components[node.id].defaults, node.config)
            for node in workflow.nodes
        }

        workflow_path = run_dir / "workflow.yaml"
        resolved_path = run_dir / "resolved_config.yaml"
        gin_path = run_dir / "generated.gin"
        command_path = run_dir / "command.sh"

        workflow_path.write_text(yaml.safe_dump(workflow.model_dump(), sort_keys=True), encoding="utf-8")
        resolved_path.write_text(yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8")
        self.gin_writer.write(gin_path, components, resolved_config)
        command = self._command(workflow, components, command_path)
        command_path.write_text(command, encoding="utf-8")
        self._make_executable(command_path)
        manifest = self._manifest(
            workflow=workflow,
            experiment_id=experiment_id,
            run_dir=run_dir,
            workflow_path=workflow_path,
            resolved_path=resolved_path,
            gin_path=gin_path,
            command_path=command_path,
        )
        manifest_path = write_manifest(run_dir, manifest)
        update_status(
            run_dir,
            RunStatusState.compiled,
            backend=workflow.execution.backend,
        )
        status_path = run_dir / "status.json"

        generated_files = [
            str(workflow_path),
            str(resolved_path),
            str(gin_path),
            str(command_path),
            str(manifest_path),
            str(status_path),
        ]

        experiment = ExperimentSpec(
            experiment_id=experiment_id,
            workflow=workflow,
            resolved_config=resolved_config,
            run_dir=str(run_dir),
            command=str(command_path),
            generated_files=generated_files,
            execution_backend=workflow.execution.backend,
        )
        if workflow.execution.backend == "slurm":
            script = SlurmExecutor.render_script(experiment, workflow.execution.options)
            slurm_path = run_dir / "slurm_job.sh"
            slurm_path.write_text(script, encoding="utf-8")
            self._make_executable(slurm_path)
            experiment.generated_files.append(str(slurm_path))
        return experiment

    def _command(
        self,
        workflow: WorkflowSpec,
        components: dict[str, ComponentSpec],
        command_path: Path,
    ) -> str:
        module = self._runner_command_module(workflow, components)
        run_dir = command_path.parent.resolve()
        project_root = Path.cwd().resolve()
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"RUN_DIR={shlex.quote(str(run_dir))}",
                f"PROJECT_ROOT={shlex.quote(str(project_root))}",
                'BACKEND="${RLFLOW_BACKEND:-local}"',
                'EXTERNAL_ID="${RLFLOW_EXTERNAL_ID:-${SLURM_JOB_ID:-}}"',
                'if [[ -z "${RLFLOW_EXTERNAL_ID:-}" && -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then',
                '  EXTERNAL_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
                "fi",
                'PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"',
                'if [[ -x "$PYTHON_BIN" ]]; then',
                '  STATUS_PY=("$PYTHON_BIN")',
                '  RUNNER_PY=("$PYTHON_BIN")',
                'elif command -v uv >/dev/null 2>&1; then',
                '  cd "$PROJECT_ROOT"',
                "  STATUS_PY=(uv run python)",
                "  RUNNER_PY=(uv run python)",
                "else",
                "  STATUS_PY=(python3)",
                "  RUNNER_PY=(python3)",
                "fi",
                '"${STATUS_PY[@]}" -m rlflow.tracking.mark_status --run-dir "$RUN_DIR" running --backend "$BACKEND" --external-id "$EXTERNAL_ID" || true',
                "set +e",
                (
                    f'"${{RUNNER_PY[@]}}" -m {module} --workflow "$RUN_DIR/workflow.yaml" '
                    f'--gin_file "$RUN_DIR/generated.gin" --resolved_config "$RUN_DIR/resolved_config.yaml" '
                    '--run_dir "$RUN_DIR"'
                ),
                "EXIT_CODE=$?",
                "set -e",
                'if [[ "$EXIT_CODE" -eq 0 ]]; then',
                '  "${STATUS_PY[@]}" -m rlflow.tracking.mark_status --run-dir "$RUN_DIR" completed --exit-code "$EXIT_CODE" --backend "$BACKEND" --external-id "$EXTERNAL_ID" || true',
                "else",
                '  "${STATUS_PY[@]}" -m rlflow.tracking.mark_status --run-dir "$RUN_DIR" failed --exit-code "$EXIT_CODE" --backend "$BACKEND" --external-id "$EXTERNAL_ID" || true',
                "fi",
                'exit "$EXIT_CODE"',
                "",
            ]
        )

    def _runner_command_module(
        self,
        workflow: WorkflowSpec,
        components: dict[str, ComponentSpec],
    ) -> str:
        for node in workflow.nodes:
            component = components[node.id]
            if component.kind == "runner":
                return component.compile_target.get("command", {}).get(
                    "module",
                    "rlflow_builtin.runners.tabular_jax",
                )
        return "rlflow_builtin.runners.tabular_jax"

    def _merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _experiment_id(self, name: str) -> str:
        return make_run_id(name)

    def _make_executable(self, path: Path) -> None:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _create_run_directories(self, run_dir: Path) -> None:
        for path in [
            run_dir / "logs",
            run_dir / "summaries",
            run_dir / "artifacts",
            run_dir / "artifacts" / "checkpoints",
            run_dir / "artifacts" / "replay",
            run_dir / "artifacts" / "arrays",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _manifest(
        self,
        *,
        workflow: WorkflowSpec,
        experiment_id: str,
        run_dir: Path,
        workflow_path: Path,
        resolved_path: Path,
        gin_path: Path,
        command_path: Path,
    ) -> RunManifest:
        metadata = workflow.metadata
        git_commit, git_dirty = collect_git_info(Path.cwd())
        return RunManifest(
            run_id=experiment_id,
            experiment_id=experiment_id,
            sweep_id=metadata.get("sweep_id"),
            sweep_group_id=metadata.get("sweep_group_id"),
            sweep_trial_id=metadata.get("sweep_trial_id"),
            seed=metadata.get("seed"),
            created_at=utc_now_iso(),
            run_dir=str(run_dir),
            workflow_path=str(workflow_path),
            resolved_config_path=str(resolved_path),
            generated_gin_path=str(gin_path),
            command_path=str(command_path),
            workflow_sha256=file_sha256(workflow_path),
            resolved_config_sha256=file_sha256(resolved_path),
            generated_gin_sha256=file_sha256(gin_path),
            command_sha256=file_sha256(command_path),
            git_commit=git_commit,
            git_dirty=git_dirty,
            python_version=python_version(),
            platform=platform_string(),
            dependencies=collect_dependency_versions(),
            backend=workflow.execution.backend,
        )
