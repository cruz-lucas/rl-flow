from __future__ import annotations

import copy
import csv
import html
import json
import math
import random
import re
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


_TRAIN_RETURN_LAST_RE = re.compile(r"^mean_train_return_last_(\d+)$")


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
        slurm_array_task_count = self._slurm_array_task_count(len(trial_values), spec.slurm.trials_per_task)
        if base_workflow.execution.backend == "slurm" and spec.slurm.max_array_tasks is not None:
            if slurm_array_task_count > spec.slurm.max_array_tasks:
                raise SweepCompilationError(
                    "Sweep requires "
                    f"{slurm_array_task_count} SLURM array tasks for {len(trial_values)} trials "
                    f"with trials_per_task={spec.slurm.trials_per_task}, exceeding "
                    f"max_array_tasks={spec.slurm.max_array_tasks}. Increase slurm.trials_per_task, "
                    "reduce the sweep, or set slurm.max_array_tasks."
                )

        trials: list[SweepTrial] = []
        generated_files: list[str] = []
        group_ids: dict[str, str] = {}
        for index, parameters in enumerate(trial_values):
            trial_id = f"trial-{index:04d}"
            group_id, group_run_dir, seed_value = self._trial_group(sweep_dir, parameters, group_ids)
            experiment_id = slugify_run_name(f"{sweep_id}-{trial_id}")
            workflow = self._trial_workflow(
                base_workflow,
                spec=spec,
                sweep_id=sweep_id,
                trial_id=trial_id,
                group_id=group_id,
                group_run_dir=group_run_dir,
                experiment_id=experiment_id,
                parameters=parameters,
                seed_value=seed_value,
            )
            trial_dir = self._trial_dir(sweep_dir, trial_id, parameters, group_id, group_run_dir)
            experiment = self.workflow_compiler.compile(workflow, out_dir=trial_dir)
            generated_files.extend(experiment.generated_files)
            trials.append(
                SweepTrial(
                    index=index,
                    trial_id=trial_id,
                    group_id=group_id,
                    group_run_dir=str(group_run_dir) if group_run_dir is not None else None,
                    seed_value=seed_value,
                    experiment_id=experiment.experiment_id,
                    parameters=parameters,
                    run_dir=experiment.run_dir,
                    command=experiment.command,
                    workflow_path=str(trial_dir / "workflow.yaml"),
                    metrics_path=str(trial_dir / "summaries" / "metrics.json"),
                )
            )

        compilation = SweepCompilation(
            sweep_id=sweep_id,
            name=spec.name,
            method=spec.method,
            metric=spec.metric,
            sweep_dir=str(sweep_dir),
            manifest_path=str(sweep_dir / "sweep_manifest.yaml"),
            slurm_trials_per_task=spec.slurm.trials_per_task,
            slurm_array_task_count=(
                slurm_array_task_count
                if base_workflow.execution.backend == "slurm"
                else None
            ),
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
                trials_per_task=spec.slurm.trials_per_task,
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

    def _slurm_array_task_count(self, trial_count: int, trials_per_task: int) -> int:
        return math.ceil(trial_count / trials_per_task)

    def summarize(
        self,
        manifest_path: str | Path,
        *,
        metric: str | None = None,
        goal: str | None = None,
        metric_last_n: int | None = None,
    ) -> dict[str, Any]:
        manifest_data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
        compilation = SweepCompilation.model_validate(manifest_data)
        metric_name = metric or compilation.metric.name
        metric_goal = goal or compilation.metric.goal
        resolved_last_n = metric_last_n or compilation.metric.last_n
        rows = []
        for trial in compilation.trials:
            value = self._trial_metric_value(trial, metric_name, resolved_last_n)
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "group_id": trial.group_id,
                    "seed_value": trial.seed_value,
                    "experiment_id": trial.experiment_id,
                    "run_dir": trial.run_dir,
                    "parameters": trial.parameters,
                    "metric": value,
                }
            )

        groups = self._metric_groups(rows, metric_goal)
        best = copy.deepcopy(groups[0]) if groups else None
        reverse = metric_goal == "maximize"
        completed = [row for row in rows if isinstance(row["metric"], (int, float))]
        completed.sort(key=lambda row: float(row["metric"]), reverse=reverse)
        return {
            "sweep_id": compilation.sweep_id,
            "metric": metric_name,
            "goal": metric_goal,
            "metric_last_n": resolved_last_n,
            "best": best,
            "groups": groups,
            "trials": rows,
        }

    def export_learning_curves(
        self,
        manifest_path: str | Path,
        *,
        out_dir: str | Path | None = None,
        history: str = "train",
        value: str = "discounted_return",
        bootstrap_samples: int = 1000,
        seed: int = 0,
    ) -> dict[str, Any]:
        manifest_path = Path(manifest_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "sweep_manifest.yaml"
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        compilation = SweepCompilation.model_validate(manifest_data)
        output_dir = Path(out_dir) if out_dir is not None else Path(compilation.sweep_dir) / "learning_curves"
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for trial in compilation.trials:
            series = self._history_series(Path(trial.run_dir), history=history, value=value)
            if series:
                rows.append(
                    {
                        "trial_id": trial.trial_id,
                        "parameters": trial.parameters,
                        "series": series,
                    }
                )

        groups = self._curve_groups(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        csv_path = output_dir / f"{history}_{value}_curves.csv"
        svg_path = output_dir / f"{history}_{value}_curves.svg"
        self._write_curve_csv(csv_path, groups)
        self._write_curve_svg(svg_path, groups, title=f"{history} {value}".replace("_", " ").title())
        return {
            "sweep_id": compilation.sweep_id,
            "history": history,
            "value": value,
            "bootstrap_samples": bootstrap_samples,
            "csv_path": str(csv_path),
            "svg_path": str(svg_path),
            "groups": [
                {
                    "group_id": group["group_id"],
                    "parameters": group["parameters"],
                    "seed_count": group["seed_count"],
                    "points": len(group["points"]),
                }
                for group in groups
            ],
        }

    def _metric_groups(
        self,
        rows: list[dict[str, Any]],
        metric_goal: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            metric = row["metric"]
            if not isinstance(metric, (int, float)):
                continue
            parameters = self._non_seed_parameters(row["parameters"])
            key = json.dumps(parameters, sort_keys=True, default=str)
            group = grouped.setdefault(
                key,
                {
                    "group_id": row.get("group_id") or f"group-{len(grouped):04d}",
                    "parameters": parameters,
                    "metrics": [],
                    "trial_ids": [],
                    "run_dirs": [],
                },
            )
            group["metrics"].append(float(metric))
            group["trial_ids"].append(row["trial_id"])
            group["run_dirs"].append(row["run_dir"])

        groups: list[dict[str, Any]] = []
        for group in grouped.values():
            metrics = group.pop("metrics")
            metric_mean = sum(metrics) / len(metrics)
            metric_min = min(metrics)
            metric_max = max(metrics)
            if len(metrics) > 1:
                metric_std = math.sqrt(
                    sum((value - metric_mean) ** 2 for value in metrics) / (len(metrics) - 1)
                )
            else:
                metric_std = 0.0
            groups.append(
                {
                    **group,
                    "metric": metric_mean,
                    "metric_mean": metric_mean,
                    "metric_min": metric_min,
                    "metric_max": metric_max,
                    "metric_std": metric_std,
                    "metric_count": len(metrics),
                }
            )

        reverse = metric_goal == "maximize"
        groups.sort(key=lambda group: float(group["metric_mean"]), reverse=reverse)
        return groups

    def _non_seed_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in parameters.items()
            if not self._is_seed_parameter(key)
        }

    def _is_seed_parameter(self, key: str) -> bool:
        normalized = key.lower()
        return normalized == "seed" or normalized.endswith("_seed") or normalized.endswith(".seed")

    def _trial_group(
        self,
        sweep_dir: Path,
        parameters: dict[str, Any],
        group_ids: dict[str, str],
    ) -> tuple[str | None, Path | None, Any | None]:
        seed_parameters = {
            key: value
            for key, value in parameters.items()
            if self._is_seed_parameter(key)
        }
        if not seed_parameters:
            return None, None, None

        group_parameters = self._non_seed_parameters(parameters)
        key = json.dumps(group_parameters, sort_keys=True, default=str)
        group_id = group_ids.setdefault(key, f"group-{len(group_ids):04d}")
        seed_value = (
            next(iter(seed_parameters.values()))
            if len(seed_parameters) == 1
            else seed_parameters
        )
        return group_id, sweep_dir / "trials" / group_id, seed_value

    def _trial_dir(
        self,
        sweep_dir: Path,
        trial_id: str,
        parameters: dict[str, Any],
        group_id: str | None,
        group_run_dir: Path | None,
    ) -> Path:
        if group_id is None or group_run_dir is None:
            return sweep_dir / "trials" / trial_id
        return group_run_dir / self._seed_dir_name(parameters, trial_id)

    def _seed_dir_name(self, parameters: dict[str, Any], trial_id: str) -> str:
        seed_parameters = [
            (key, value)
            for key, value in parameters.items()
            if self._is_seed_parameter(key)
        ]
        if not seed_parameters:
            return trial_id
        parts = [
            f"{slugify_run_name(key)}-{slugify_run_name(str(value))}"
            for key, value in seed_parameters
        ]
        return "-".join(parts) or trial_id

    def _trial_metric_value(
        self,
        trial: SweepTrial,
        metric_name: str,
        metric_last_n: int | None,
    ) -> float | int | None:
        metrics = self._read_trial_metrics(trial)
        if metric_name in metrics:
            value = metrics[metric_name]
            return value if isinstance(value, (int, float)) else None
        if metric_name == "mean_train_return":
            return self._mean_train_return(Path(trial.run_dir), None)
        if metric_name == "mean_train_return_last_n":
            return self._mean_train_return(Path(trial.run_dir), metric_last_n or 10)
        match = _TRAIN_RETURN_LAST_RE.match(metric_name)
        if match is not None:
            return self._mean_train_return(Path(trial.run_dir), int(match.group(1)))
        return None

    def _read_trial_metrics(self, trial: SweepTrial) -> dict[str, Any]:
        candidates = [
            Path(trial.metrics_path),
            Path(trial.run_dir) / "summaries" / "metrics.json",
            Path(trial.run_dir) / "metrics.json",
        ]
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _mean_train_return(self, run_dir: Path, count: int | None) -> float | None:
        history_path = run_dir / "logs" / "train_history.jsonl"
        if not history_path.exists():
            return None
        returns: list[float] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("return")
            if isinstance(value, (int, float)):
                returns.append(float(value))
        if not returns:
            return None
        window = returns[-count:] if count is not None else returns
        return sum(window) / len(window)

    def _history_series(
        self,
        run_dir: Path,
        *,
        history: str,
        value: str,
    ) -> list[tuple[int, float]]:
        if history not in {"train", "eval"}:
            raise SweepCompilationError("--history must be train or eval")
        history_path = run_dir / "logs" / f"{history}_history.jsonl"
        if not history_path.exists():
            return []
        fallback_value = "return" if value == "discounted_return" else value
        series: list[tuple[int, float]] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_value = row.get(value, row.get(fallback_value))
            episode = row.get("episode")
            if isinstance(raw_value, (int, float)) and isinstance(episode, int):
                series.append((episode, float(raw_value)))
        return series

    def _curve_groups(
        self,
        rows: list[dict[str, Any]],
        *,
        bootstrap_samples: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            parameters = self._non_seed_parameters(row["parameters"])
            key = json.dumps(parameters, sort_keys=True, default=str)
            group = grouped.setdefault(
                key,
                {
                    "group_id": f"group-{len(grouped):04d}",
                    "parameters": parameters,
                    "series": [],
                },
            )
            group["series"].append(row["series"])

        rng = random.Random(seed)
        groups = []
        for group in grouped.values():
            series_list = group.pop("series")
            points = self._aggregate_curve_points(
                series_list,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            groups.append(
                {
                    **group,
                    "seed_count": len(series_list),
                    "points": points,
                }
            )
        groups.sort(key=lambda group: json.dumps(group["parameters"], sort_keys=True, default=str))
        return groups

    def _aggregate_curve_points(
        self,
        series_list: list[list[tuple[int, float]]],
        *,
        bootstrap_samples: int,
        rng: random.Random,
    ) -> list[dict[str, float | int]]:
        episode_values: dict[int, list[float]] = {}
        for series in series_list:
            for episode, value in series:
                episode_values.setdefault(episode, []).append(value)

        points: list[dict[str, float | int]] = []
        for episode in sorted(episode_values):
            values = episode_values[episode]
            mean = sum(values) / len(values)
            if len(values) == 1 or bootstrap_samples <= 0:
                ci_low = mean
                ci_high = mean
            else:
                means = []
                for _ in range(bootstrap_samples):
                    sample = [values[rng.randrange(len(values))] for _ in values]
                    means.append(sum(sample) / len(sample))
                means.sort()
                low_index = min(len(means) - 1, max(0, int(0.025 * len(means))))
                high_index = min(len(means) - 1, max(0, int(0.975 * len(means))))
                ci_low = means[low_index]
                ci_high = means[high_index]
            points.append(
                {
                    "episode": episode,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "seed_count": len(values),
                }
            )
        return points

    def _write_curve_csv(self, path: Path, groups: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "group_id",
                    "episode",
                    "mean",
                    "ci_low",
                    "ci_high",
                    "seed_count",
                    "parameters",
                ],
            )
            writer.writeheader()
            for group in groups:
                parameters = json.dumps(group["parameters"], sort_keys=True, default=str)
                for point in group["points"]:
                    writer.writerow(
                        {
                            "group_id": group["group_id"],
                            "episode": point["episode"],
                            "mean": point["mean"],
                            "ci_low": point["ci_low"],
                            "ci_high": point["ci_high"],
                            "seed_count": point["seed_count"],
                            "parameters": parameters,
                        }
                    )

    def _write_curve_svg(self, path: Path, groups: list[dict[str, Any]], *, title: str) -> None:
        drawable_groups = [group for group in groups if group["points"]]
        width = 900
        height = 520
        margin_left = 70
        margin_right = 24
        margin_top = 54
        margin_bottom = 64
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        all_points = [point for group in drawable_groups for point in group["points"]]
        if not all_points:
            path.write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>',
                encoding="utf-8",
            )
            return
        x_min = min(int(point["episode"]) for point in all_points)
        x_max = max(int(point["episode"]) for point in all_points)
        y_min = min(float(point["ci_low"]) for point in all_points)
        y_max = max(float(point["ci_high"]) for point in all_points)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1.0
        y_pad = 0.05 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad

        def sx(value: float) -> float:
            return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

        def sy(value: float) -> float:
            return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

        colors = ["#0f5f6f", "#a23b72", "#2f7d32", "#c05621", "#4c51bf", "#6b7280"]
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{margin_left}" y="30" font-family="Arial" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151"/>',
            f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">Episode</text>',
            f'<text x="18" y="{margin_top + plot_height / 2}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151" transform="rotate(-90 18 {margin_top + plot_height / 2})">Discounted return</text>',
        ]
        for index, group in enumerate(drawable_groups):
            color = colors[index % len(colors)]
            points = group["points"]
            upper = [(sx(float(point["episode"])), sy(float(point["ci_high"]))) for point in points]
            lower = [(sx(float(point["episode"])), sy(float(point["ci_low"]))) for point in reversed(points)]
            band = " ".join(f"{x:.2f},{y:.2f}" for x, y in [*upper, *lower])
            line = " ".join(
                f'{sx(float(point["episode"])):.2f},{sy(float(point["mean"])):.2f}'
                for point in points
            )
            label = html.escape(json.dumps(group["parameters"], sort_keys=True, default=str) or "{}")
            parts.extend(
                [
                    f'<polygon points="{band}" fill="{color}" opacity="0.16"/>',
                    f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>',
                    f'<text x="{margin_left + 12}" y="{margin_top + 18 + index * 18}" font-family="Arial" font-size="11" fill="{color}">{group["group_id"]}: {label}</text>',
                ]
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

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
        group_id: str | None,
        group_run_dir: Path | None,
        experiment_id: str,
        parameters: dict[str, Any],
        seed_value: Any | None,
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
        if group_id is not None:
            data["metadata"]["sweep_group_id"] = group_id
            data["metadata"]["sweep_group_run_dir"] = str(group_run_dir)
            data["metadata"]["sweep_group_parameters"] = self._non_seed_parameters(parameters)
            data["metadata"]["seed"] = seed_value
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
