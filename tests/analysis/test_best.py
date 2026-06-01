import json
from pathlib import Path

import yaml

from rlflow.analysis.best import export_best_config
from rlflow.schemas.sweep import SweepCompilation


def test_export_best_config_copies_workflow_and_resolved(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial-0000"
    compiled_dir = tmp_path / "compiled-trial-0000"
    trial_dir.mkdir(parents=True)
    compiled_dir.mkdir(parents=True)
    workflow_path = compiled_dir / "workflow.yaml"
    resolved_path = compiled_dir / "resolved_config.yaml"
    workflow_path.write_text(yaml.safe_dump({"name": "test"}), encoding="utf-8")
    resolved_path.write_text(yaml.safe_dump({"config": 1}), encoding="utf-8")

    manifest = SweepCompilation.model_validate(
        {
            "sweep_id": "best-sweep",
            "name": "best sweep",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "sweep_dir": str(tmp_path),
            "manifest_path": str(tmp_path / "sweep_manifest.yaml"),
            "trials": [
                {
                    "index": 0,
                    "trial_id": "trial-0000",
                    "group_id": "group-0000",
                    "seed_value": 0,
                    "experiment_id": "exp-0",
                    "parameters": {"lr": 0.1, "seed": 0},
                    "run_dir": str(trial_dir),
                    "command": "",
                    "workflow_path": str(workflow_path),
                    "metrics_path": str(trial_dir / "metrics.json"),
                }
            ],
        }
    )
    (tmp_path / "sweep_manifest.yaml").write_text(manifest.model_dump_json(), encoding="utf-8")
    Path(trial_dir / "metrics.json").write_text(json.dumps({"mean_eval_return": 5.0}), encoding="utf-8")

    output = export_best_config(tmp_path / "sweep_manifest.yaml", metric="mean_eval_return", goal="maximize")
    assert Path(output["best_group"]).exists()
    assert Path(output["best_workflow"]).exists()
    assert Path(output["best_resolved_config"]).exists()
    assert yaml.safe_load(Path(output["best_workflow"]).read_text(encoding="utf-8"))["name"] == "test"
    assert yaml.safe_load(Path(output["best_resolved_config"]).read_text(encoding="utf-8"))["config"] == 1
