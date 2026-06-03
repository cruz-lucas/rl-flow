import json
from pathlib import Path

from rlflow.analysis.report import export_sweep_report, format_sweep_report


def _summary() -> dict:
    return {
        "sweep_id": "dqn-smoke",
        "metric": "mean_train_return_last_n",
        "goal": "maximize",
        "metric_last_n": 50,
        "groups": [
            {
                "group_id": "group-0000",
                "parameters": {"batch_size": 32, "learning_rate": 0.0001},
                "metric_mean": 0.9,
                "metric_std": 0.0,
                "metric_min": 0.9,
                "metric_max": 0.9,
                "metric_count": 2,
                "trial_ids": ["trial-0000", "trial-0001"],
                "run_dirs": ["/runs/seed-0", "/runs/seed-1"],
            },
            {
                "group_id": "group-0001",
                "parameters": {"batch_size": 64, "learning_rate": 0.0001},
                "metric_mean": 0.1,
                "metric_std": 0.2,
                "metric_min": 0.0,
                "metric_max": 0.2,
                "metric_count": 2,
                "trial_ids": ["trial-0002", "trial-0003"],
                "run_dirs": ["/runs/seed-0", "/runs/seed-1"],
            },
        ],
        "trials": [
            {
                "trial_id": "trial-0000",
                "group_id": "group-0000",
                "seed_value": 0,
                "parameters": {"batch_size": 32, "learning_rate": 0.0001, "seed": 0},
                "metric": 0.9,
                "run_dir": "/runs/seed-0",
            },
            {
                "trial_id": "trial-0001",
                "group_id": "group-0000",
                "seed_value": 1,
                "parameters": {"batch_size": 32, "learning_rate": 0.0001, "seed": 1},
                "metric": 0.9,
                "run_dir": "/runs/seed-1",
            },
            {
                "trial_id": "trial-0002",
                "group_id": "group-0001",
                "seed_value": 0,
                "parameters": {"batch_size": 64, "learning_rate": 0.0001, "seed": 0},
                "metric": None,
                "run_dir": "/runs/seed-0",
            },
        ],
    }


def test_format_sweep_report_prints_ranked_groups() -> None:
    report = format_sweep_report(_summary(), top_k=1)

    assert "Sweep: dqn-smoke" in report
    assert "Metric: mean_train_return_last_n (maximize)" in report
    assert "Trials: 2/3 completed, 1 missing metric" in report
    assert "group-0000" in report
    assert "batch_size=32, learning_rate=0.0001" in report
    assert "group-0001" not in report
    assert "pass --all" in report


def test_format_sweep_report_can_include_trials() -> None:
    report = format_sweep_report(_summary(), top_k=None, include_trials=True)

    assert "Trials\n" in report
    assert "trial-0000" in report
    assert "seed=0" in report


def test_export_sweep_report_writes_terminal_report_json_and_csv(tmp_path: Path) -> None:
    paths = export_sweep_report(_summary(), out_dir=tmp_path)

    assert Path(paths["report"]).read_text(encoding="utf-8").startswith("Sweep: dqn-smoke")
    data = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert data["sweep_id"] == "dqn-smoke"
    csv_text = Path(paths["groups_csv"]).read_text(encoding="utf-8")
    assert "group-0000" in csv_text
    assert '"{""batch_size"": 32, ""learning_rate"": 0.0001}"' in csv_text
