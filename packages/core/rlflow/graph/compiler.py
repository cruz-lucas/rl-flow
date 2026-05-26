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
        (run_dir / "logs").mkdir(exist_ok=True)

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

        generated_files = [
            str(workflow_path),
            str(resolved_path),
            str(gin_path),
            str(command_path),
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
                'PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"',
                'if [[ -x "$PYTHON_BIN" ]]; then',
                (
                    f'  "$PYTHON_BIN" -m {module} --workflow "$RUN_DIR/workflow.yaml" '
                    f'--gin_file "$RUN_DIR/generated.gin" --resolved_config "$RUN_DIR/resolved_config.yaml" '
                    '--run_dir "$RUN_DIR"'
                ),
                'elif command -v uv >/dev/null 2>&1; then',
                '  cd "$PROJECT_ROOT"',
                (
                    f'  uv run python -m {module} --workflow "$RUN_DIR/workflow.yaml" '
                    f'--gin_file "$RUN_DIR/generated.gin" --resolved_config "$RUN_DIR/resolved_config.yaml" '
                    '--run_dir "$RUN_DIR"'
                ),
                "else",
                (
                    f'  python3 -m {module} --workflow "$RUN_DIR/workflow.yaml" '
                    f'--gin_file "$RUN_DIR/generated.gin" --resolved_config "$RUN_DIR/resolved_config.yaml" '
                    '--run_dir "$RUN_DIR"'
                ),
                "fi",
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
