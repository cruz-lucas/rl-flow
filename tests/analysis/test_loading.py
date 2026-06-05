from pathlib import Path

import pandas as pd

from rlflow.analysis.loading import load_sweep_histories, load_sweep_manifest, non_seed_parameters
from rlflow.schemas.sweep import SweepCompilation, SweepSpec


def test_load_sweep_histories_skips_missing_files(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(
        {
            "name": "history sweep",
            "sweep_id": "history-sweep",
            "workflow": "configs/workflows/tabular_q_learning_riverswim.yaml",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "parameters": {
                "seed": {
                    "target": "nodes.runner.config.seed",
                    "values": [0, 1],
                },
            },
        }
    )
    # create a fake manifest and one history file
    sweep_dir = tmp_path
    trial_run_dir = sweep_dir / "trial-0000"
    trial_run_dir.mkdir(parents=True)
    (trial_run_dir / "logs").mkdir()
    (trial_run_dir / "logs" / "train_history.jsonl").write_text(
        '{"episode": 0, "return": 1.0} \n', encoding="utf-8"
    )
    manifest = SweepCompilation.model_validate(
        {
            "sweep_id": "history-sweep",
            "name": "history sweep",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "sweep_dir": str(sweep_dir),
            "manifest_path": str(sweep_dir / "sweep_manifest.yaml"),
            "trials": [
                {
                    "index": 0,
                    "trial_id": "trial-0000",
                    "group_id": None,
                    "seed_value": 0,
                    "experiment_id": "exp-0",
                    "parameters": {"seed": 0},
                    "run_dir": str(trial_run_dir),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": "",
                },
                {
                    "index": 1,
                    "trial_id": "trial-0001",
                    "group_id": None,
                    "seed_value": 1,
                    "experiment_id": "exp-1",
                    "parameters": {"seed": 1},
                    "run_dir": str(sweep_dir / "trial-0001"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": "",
                },
            ],
        }
    )
    manifest_path = sweep_dir / "sweep_manifest.yaml"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    df = load_sweep_histories(manifest_path, history="train")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["trial_id"] == "trial-0000"


def test_load_sweep_manifest_rebases_paths_when_sweep_is_moved(tmp_path: Path) -> None:
    original_sweep_dir = tmp_path / "original-sweep"
    moved_sweep_dir = tmp_path / "moved-sweep"
    original_run_dir = original_sweep_dir / "trials" / "group-0000" / "seed-0"
    moved_run_dir = moved_sweep_dir / "trials" / "group-0000" / "seed-0"
    (moved_run_dir / "logs").mkdir(parents=True)
    (moved_run_dir / "logs" / "train_history.jsonl").write_text(
        '{"episode": 0, "return": 1.0} \n',
        encoding="utf-8",
    )
    manifest = SweepCompilation.model_validate(
        {
            "sweep_id": "moved-sweep",
            "name": "moved sweep",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "sweep_dir": str(original_sweep_dir),
            "manifest_path": str(original_sweep_dir / "sweep_manifest.yaml"),
            "slurm_array_path": str(original_sweep_dir / "slurm_array.sh"),
            "generated_files": [str(original_run_dir / "workflow.yaml")],
            "trials": [
                {
                    "index": 0,
                    "trial_id": "trial-0000",
                    "group_id": "group-0000",
                    "group_run_dir": str(original_run_dir.parent),
                    "seed_value": 0,
                    "experiment_id": "exp-0",
                    "parameters": {"seed": 0},
                    "run_dir": str(original_run_dir),
                    "command": str(original_run_dir / "command.sh"),
                    "workflow_path": str(original_run_dir / "workflow.yaml"),
                    "metrics_path": str(original_run_dir / "metrics.json"),
                }
            ],
        }
    )
    manifest_path = moved_sweep_dir / "sweep_manifest.yaml"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    loaded = load_sweep_manifest(manifest_path)
    trial = loaded.trials[0]

    assert loaded.sweep_dir == str(moved_sweep_dir)
    assert loaded.manifest_path == str(manifest_path)
    assert loaded.slurm_array_path == str(moved_sweep_dir / "slurm_array.sh")
    assert loaded.generated_files == [str(moved_run_dir / "workflow.yaml")]
    assert trial.group_run_dir == str(moved_run_dir.parent)
    assert trial.run_dir == str(moved_run_dir)
    assert trial.command == str(moved_run_dir / "command.sh")
    assert trial.workflow_path == str(moved_run_dir / "workflow.yaml")
    assert trial.metrics_path == str(moved_run_dir / "metrics.json")

    df = load_sweep_histories(manifest_path, history="train")
    assert len(df) == 1
    assert df.iloc[0]["run_dir"] == str(moved_run_dir)


def test_non_seed_parameters_removes_seed_keys() -> None:
    original = {"seed": 0, "learning_rate": 0.1, "env.seed": 42, "batch_seed": 128}
    result = non_seed_parameters(original)
    assert result == {"learning_rate": 0.1}
