from __future__ import annotations

import json
import math
import random
import stat
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from rlflow.execution.slurm import SlurmExecutor
from rlflow.graph.compiler import WorkflowCompiler
from rlflow.graph.run_naming import make_run_id, slugify_run_name
from rlflow.registry.base import ComponentRegistry
from rlflow.schemas.sweep import SweepCompilation, SweepParameter, SweepSpec, SweepTrial
from rlflow.schemas.workflow import WorkflowSpec


class SweepCompilationError(ValueError):
    pass


class SweepCompiler:
    def __init__(self, registry: ComponentRegistry) -> None:
        self.workflow_compiler = WorkflowCompiler(registry)

    def compile(
        self,
        spec: SweepSpec,
        *,
        base_path: str | Path | None = None,
        out_dir: str | Path | None = None,
    ) -> SweepCompilation:
        sweep_id = spec.sweep_id or make_run_id(spec.name)
        sweep_dir = (
            Path(out_dir)
            if out_dir
            else Path("runs") / "sweeps" / slugify_run_name(spec.name) / sweep_id
        )
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_dir = sweep_dir.resolve()
        (sweep_dir / "logs").mkdir(exist_ok=True)
        (sweep_dir / "trials").mkdir(exist_ok=True)

        base_workflow = self._load_workflow(spec.workflow, base_path=base_path)
        if spec.execution is not None:
            base_workflow.execution = spec.execution

        trial_values = self._expand_trials(spec)
        if not trial_values:
            raise SweepCompilationError("Sweep did not produce any trials")

        trials: list[SweepTrial] = []
        generated_files: list[str] = []
        for index, parameters in enumerate(trial_values):
            trial_id = f"trial-{index:04d}"
            experiment_id = slugify_run_name(f"{sweep_id}-{trial_id}")
            workflow = self._trial_workflow(
                base_workflow,
                spec=spec,
                sweep_id=sweep_id,
                trial_id=trial_id,
                experiment_id=experiment_id,
                parameters=parameters,
            )
            trial_dir = sweep_dir / "trials" / trial_id
            experiment = self.workflow_compiler.compile(workflow, out_dir=trial_dir)
            generated_files.extend(experiment.generated_files)
            trials.append(
                SweepTrial(
                    index=index,
                    trial_id=trial_id,
                    experiment_id=experiment.experiment_id,
                    parameters=parameters,
                    run_dir=experiment.run_dir,
                    command=experiment.command,
                    workflow_path=str(trial_dir / "workflow.yaml"),
                    metrics_path=str(trial_dir / "metrics.json"),
                )
            )

        compilation = SweepCompilation(
            sweep_id=sweep_id,
            name=spec.name,
            method=spec.method,
            metric=spec.metric,
            sweep_dir=str(sweep_dir),
            manifest_path=str(sweep_dir / "sweep_manifest.yaml"),
            trials=trials,
            generated_files=generated_files,
        )

        manifest_path = Path(compilation.manifest_path)
        manifest_path.write_text(
            yaml.safe_dump(compilation.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        compilation.generated_files.append(str(manifest_path))

        if base_workflow.execution.backend == "slurm":
            slurm_path = sweep_dir / "slurm_array.sh"
            script = SlurmExecutor.render_array_script(
                compilation,
                base_workflow.execution.options,
                max_parallel=spec.slurm.max_parallel,
            )
            slurm_path.write_text(script, encoding="utf-8")
            self._make_executable(slurm_path)
            compilation.slurm_array_path = str(slurm_path)
            compilation.generated_files.append(str(slurm_path))
            manifest_path.write_text(
                yaml.safe_dump(compilation.model_dump(mode="json"), sort_keys=True),
                encoding="utf-8",
            )

        return compilation

    def summarize(
        self,
        manifest_path: str | Path,
        *,
        metric: str | None = None,
        goal: str | None = None,
    ) -> dict[str, Any]:
        manifest_data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
        compilation = SweepCompilation.model_validate(manifest_data)
        metric_name = metric or compilation.metric.name
        metric_goal = goal or compilation.metric.goal
        rows = []
        for trial in compilation.trials:
            metrics_path = Path(trial.metrics_path)
            metrics = {}
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = metrics.get(metric_name)
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "experiment_id": trial.experiment_id,
                    "run_dir": trial.run_dir,
                    "parameters": trial.parameters,
                    "metric": value,
                }
            )

        reverse = metric_goal == "maximize"
        completed = [row for row in rows if isinstance(row["metric"], (int, float))]
        completed.sort(key=lambda row: float(row["metric"]), reverse=reverse)
        best = completed[0] if completed else None
        return {
            "sweep_id": compilation.sweep_id,
            "metric": metric_name,
            "goal": metric_goal,
            "best": best,
            "trials": rows,
        }

    def _load_workflow(self, workflow: str | WorkflowSpec, *, base_path: str | Path | None) -> WorkflowSpec:
        if isinstance(workflow, WorkflowSpec):
            return workflow.model_copy(deep=True)

        path = Path(workflow)
        if not path.is_absolute() and base_path is not None:
            path = Path(base_path) / path
        if not path.exists():
            raise SweepCompilationError(f"Workflow file does not exist: {path}")
        return WorkflowSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def _expand_trials(self, spec: SweepSpec) -> list[dict[str, Any]]:
        if spec.method == "grid":
            return self._expand_grid(spec)
        return self._expand_random(spec)

    def _expand_grid(self, spec: SweepSpec) -> list[dict[str, Any]]:
        spaces: list[tuple[str, list[Any]]] = []
        for label, parameter in spec.parameters.items():
            if parameter.values is None:
                raise SweepCompilationError(f"Grid parameter {label!r} requires values")
            spaces.append((label, parameter.values))
        labels = [label for label, _ in spaces]
        value_lists = [values for _, values in spaces]
        return [
            dict(zip(labels, values, strict=True))
            for values in product(*value_lists)
        ]

    def _expand_random(self, spec: SweepSpec) -> list[dict[str, Any]]:
        rng = random.Random(spec.seed)
        count = spec.num_trials or 20
        return [
            {
                label: self._sample_parameter(parameter, rng)
                for label, parameter in spec.parameters.items()
            }
            for _ in range(count)
        ]

    def _sample_parameter(self, parameter: SweepParameter, rng: random.Random) -> Any:
        if parameter.values is not None:
            return rng.choice(parameter.values)
        if parameter.minimum is None or parameter.maximum is None:
            raise SweepCompilationError(f"Random parameter {parameter.target!r} is missing bounds")
        if parameter.distribution == "uniform":
            return rng.uniform(float(parameter.minimum), float(parameter.maximum))
        if parameter.distribution == "loguniform":
            low = math.log(float(parameter.minimum))
            high = math.log(float(parameter.maximum))
            return math.exp(rng.uniform(low, high))
        if parameter.distribution == "int_uniform":
            return rng.randint(int(parameter.minimum), int(parameter.maximum))
        raise SweepCompilationError(f"Unsupported random distribution: {parameter.distribution}")

    def _trial_workflow(
        self,
        base_workflow: WorkflowSpec,
        *,
        spec: SweepSpec,
        sweep_id: str,
        trial_id: str,
        experiment_id: str,
        parameters: dict[str, Any],
    ) -> WorkflowSpec:
        data = base_workflow.model_dump(mode="python")
        for label, value in parameters.items():
            target = spec.parameters[label].target
            self._set_target(data, target, value)
        data["metadata"] = {
            **data.get("metadata", {}),
            "experiment_id": experiment_id,
            "sweep_id": sweep_id,
            "sweep_trial_id": trial_id,
            "sweep_parameters": parameters,
        }
        return WorkflowSpec.model_validate(data)

    def _set_target(self, workflow_data: dict[str, Any], target: str, value: Any) -> None:
        parts = target.split(".")
        if len(parts) < 2:
            raise SweepCompilationError(f"Invalid sweep target: {target!r}")
        if parts[0] == "nodes":
            if len(parts) < 4:
                raise SweepCompilationError(
                    "Node sweep targets must look like nodes.<node_id>.config.<field>"
                )
            node_id = parts[1]
            node = self._node_data(workflow_data, node_id)
            self._set_nested(node, parts[2:], value)
            return
        self._set_nested(workflow_data, parts, value)

    def _node_data(self, workflow_data: dict[str, Any], node_id: str) -> dict[str, Any]:
        for node in workflow_data.get("nodes", []):
            if node.get("id") == node_id:
                return node
        raise SweepCompilationError(f"Sweep target references unknown node: {node_id}")

    def _set_nested(self, current: Any, parts: list[str], value: Any) -> None:
        for part in parts[:-1]:
            current = self._descend(current, part)
        self._assign(current, parts[-1], value)

    def _descend(self, current: Any, part: str) -> Any:
        if isinstance(current, list):
            try:
                return current[int(part)]
            except (ValueError, IndexError) as exc:
                raise SweepCompilationError(f"Invalid list index in sweep target: {part}") from exc
        if not isinstance(current, dict):
            raise SweepCompilationError(f"Cannot descend into sweep target part: {part}")
        if part not in current or current[part] is None:
            current[part] = {}
        return current[part]

    def _assign(self, current: Any, part: str, value: Any) -> None:
        if isinstance(current, list):
            try:
                current[int(part)] = value
                return
            except (ValueError, IndexError) as exc:
                raise SweepCompilationError(f"Invalid list index in sweep target: {part}") from exc
        if not isinstance(current, dict):
            raise SweepCompilationError(f"Cannot assign sweep target part: {part}")
        current[part] = value

    def _make_executable(self, path: Path) -> None:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
