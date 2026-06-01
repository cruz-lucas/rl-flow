from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from rlflow.analysis.loading import load_sweep_manifest
from rlflow.analysis.summary import summarize_groups


def export_best_config(
    manifest_path: str | Path,
    *,
    metric: str,
    goal: str,
    out_dir: str | Path | None = None,
) -> dict[str, str]:
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        sweep_dir = manifest_path
        manifest_path = manifest_path / "sweep_manifest.yaml"
    else:
        sweep_dir = manifest_path.parent

    out_dir = Path(out_dir) if out_dir else sweep_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_groups(manifest_path, metric=metric, goal=goal)
    if summary.empty:
        raise ValueError(f"No completed group found for metric: {metric}")

    compilation = load_sweep_manifest(manifest_path)
    best = summary.iloc[0].to_dict()
    trial_ids = best.get("trial_ids")
    if not trial_ids or not isinstance(trial_ids, list):
        raise ValueError("Best group has no trial ids")

    representative_trial_id = str(trial_ids[0])
    representative = next(
        (trial for trial in compilation.trials if trial.trial_id == representative_trial_id),
        None,
    )
    if representative is None:
        raise ValueError(f"Best trial not found in manifest: {representative_trial_id}")

    workflow_src = Path(representative.workflow_path)
    resolved_src = workflow_src.parent / "resolved_config.yaml"
    if not resolved_src.exists():
        resolved_src = Path(representative.run_dir) / "resolved_config.yaml"

    if not workflow_src.exists():
        raise FileNotFoundError(workflow_src)
    if not resolved_src.exists():
        raise FileNotFoundError(resolved_src)

    workflow_dst = out_dir / "best_workflow.yaml"
    resolved_dst = out_dir / "best_resolved_config.yaml"
    group_dst = out_dir / "best_group.yaml"

    shutil.copyfile(workflow_src, workflow_dst)
    shutil.copyfile(resolved_src, resolved_dst)

    group_dst.write_text(
        yaml.safe_dump(_yaml_safe(best), sort_keys=True),
        encoding="utf-8",
    )

    return {
        "best_group": str(group_dst),
        "best_workflow": str(workflow_dst),
        "best_resolved_config": str(resolved_dst),
    }


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
