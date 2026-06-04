import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from rlflow.cli import app
from rlflow.tracking.status import update_status


def test_sweep_status_reconstructs_trial_states(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    completed_dir = sweep_dir / "trials" / "group-0000" / "seed-0"
    failed_dir = sweep_dir / "trials" / "group-0000" / "seed-1"
    running_dir = sweep_dir / "trials" / "group-0001" / "seed-0"
    compiled_dir = sweep_dir / "trials" / "group-0001" / "seed-1"
    for run_dir in [completed_dir, failed_dir, running_dir, compiled_dir]:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (completed_dir / "summaries").mkdir()
    (completed_dir / "summaries" / "metrics.json").write_text(
        json.dumps({"mean_eval_return": 2.0}),
        encoding="utf-8",
    )
    update_status(completed_dir, "completed")
    update_status(failed_dir, "failed", message="boom")
    (failed_dir / "logs" / "local.err").write_text("traceback\nboom\n", encoding="utf-8")
    update_status(running_dir, "running")
    update_status(compiled_dir, "compiled")

    missing_dir = sweep_dir / "trials" / "group-0002" / "seed-0"
    manifest = {
        "sweep_id": "status-sweep",
        "name": "status sweep",
        "method": "grid",
        "metric": {"name": "mean_eval_return", "goal": "maximize"},
        "sweep_dir": str(sweep_dir),
        "manifest_path": str(sweep_dir / "sweep_manifest.yaml"),
        "slurm_array_path": None,
        "generated_files": [],
        "trials": [
            _trial(0, completed_dir, "group-0000", 0),
            _trial(1, failed_dir, "group-0000", 1),
            _trial(2, running_dir, "group-0001", 0),
            _trial(3, compiled_dir, "group-0001", 1),
            _trial(4, missing_dir, "group-0002", 0),
        ],
    }
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    result = CliRunner().invoke(app, ["sweep", "status", str(sweep_dir)])

    assert result.exit_code == 0, result.output
    assert "total trials: 5" in result.output
    assert "completed: 1" in result.output
    assert "failed: 1" in result.output
    assert "running: 1" in result.output
    assert "compiled: 1" in result.output
    assert "missing: 1" in result.output
    assert "best completed group: group-0000" in result.output
    assert "trial-0001: boom" in result.output


def _trial(index: int, run_dir: Path, group_id: str, seed: int) -> dict:
    return {
        "index": index,
        "trial_id": f"trial-{index:04d}",
        "group_id": group_id,
        "group_run_dir": str(run_dir.parent),
        "seed_value": seed,
        "experiment_id": f"exp-{index}",
        "parameters": {"group": group_id, "seed": seed},
        "run_dir": str(run_dir),
        "command": str(run_dir / "command.sh"),
        "workflow_path": str(run_dir / "workflow.yaml"),
        "metrics_path": str(run_dir / "summaries" / "metrics.json"),
    }
