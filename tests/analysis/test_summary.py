import json
from pathlib import Path

import pandas as pd
import pytest

from rlflow.analysis.summary import export_summary_tables, filter_top_k_curves, summarize_groups
from rlflow.schemas.sweep import SweepCompilation


def test_summary_ranks_groups_by_metric(tmp_path: Path) -> None:
    sweep_dir = tmp_path
    compilation = SweepCompilation.model_validate(
        {
            "sweep_id": "summary-sweep",
            "name": "summary sweep",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "sweep_dir": str(sweep_dir),
            "manifest_path": str(sweep_dir / "sweep_manifest.yaml"),
            "trials": [
                {
                    "index": 0,
                    "trial_id": "trial-0000",
                    "group_id": "group-0000",
                    "seed_value": 0,
                    "experiment_id": "exp-0",
                    "parameters": {"lr": 0.1, "seed": 0},
                    "run_dir": str(sweep_dir / "trial-0000"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(sweep_dir / "trial-0000" / "metrics.json"),
                },
                {
                    "index": 1,
                    "trial_id": "trial-0001",
                    "group_id": "group-0000",
                    "seed_value": 1,
                    "experiment_id": "exp-1",
                    "parameters": {"lr": 0.1, "seed": 1},
                    "run_dir": str(sweep_dir / "trial-0001"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(sweep_dir / "trial-0001" / "metrics.json"),
                },
                {
                    "index": 2,
                    "trial_id": "trial-0002",
                    "group_id": "group-0001",
                    "seed_value": 0,
                    "experiment_id": "exp-2",
                    "parameters": {"lr": 0.2, "seed": 0},
                    "run_dir": str(sweep_dir / "trial-0002"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(sweep_dir / "trial-0002" / "metrics.json"),
                },
            ],
        }
    )
    for trial, metric in [(compilation.trials[0], 1.0), (compilation.trials[1], 3.0), (compilation.trials[2], 4.0)]:
        Path(trial.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trial.metrics_path).write_text(json.dumps({"mean_eval_return": metric}), encoding="utf-8")
    (sweep_dir / "sweep_manifest.yaml").write_text(compilation.model_dump_json(), encoding="utf-8")

    summary = summarize_groups(sweep_dir / "sweep_manifest.yaml", metric="mean_eval_return", goal="maximize")
    assert list(summary["rank"]) == [1, 2]
    assert summary.iloc[0]["parameters"] == {"lr": 0.2}
    assert summary.iloc[0]["metric_mean"] == 4.0
    assert summary.iloc[1]["metric_mean"] == 2.0

    minimized = summarize_groups(sweep_dir / "sweep_manifest.yaml", metric="mean_eval_return", goal="minimize")
    assert minimized.iloc[0]["parameters"] == {"lr": 0.1}


def test_summary_handles_missing_metrics_and_invalid_goal(tmp_path: Path) -> None:
    compilation = SweepCompilation.model_validate(
        {
            "sweep_id": "summary-sweep",
            "name": "summary sweep",
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
                    "run_dir": str(tmp_path / "trial-0000"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(tmp_path / "trial-0000" / "metrics.json"),
                }
            ],
        }
    )
    (tmp_path / "sweep_manifest.yaml").write_text(compilation.model_dump_json(), encoding="utf-8")

    summary = summarize_groups(tmp_path / "sweep_manifest.yaml", metric="mean_eval_return", goal="maximize")
    assert summary.empty

    with pytest.raises(ValueError, match="goal must be"):
        summarize_groups(tmp_path / "sweep_manifest.yaml", metric="mean_eval_return", goal="largest")


def test_summary_computes_train_discounted_return_last_n_from_history(tmp_path: Path) -> None:
    compilation = SweepCompilation.model_validate(
        {
            "sweep_id": "summary-sweep",
            "name": "summary sweep",
            "method": "grid",
            "metric": {"name": "mean_train_discounted_return_last_n", "goal": "maximize"},
            "sweep_dir": str(tmp_path),
            "manifest_path": str(tmp_path / "sweep_manifest.yaml"),
            "trials": [
                {
                    "index": 0,
                    "trial_id": "trial-0000",
                    "group_id": "group-0000",
                    "seed_value": 0,
                    "experiment_id": "exp-0",
                    "parameters": {"seed": 0},
                    "run_dir": str(tmp_path / "trial-0000"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(tmp_path / "trial-0000" / "metrics.json"),
                },
                {
                    "index": 1,
                    "trial_id": "trial-0001",
                    "group_id": "group-0000",
                    "seed_value": 1,
                    "experiment_id": "exp-1",
                    "parameters": {"seed": 1},
                    "run_dir": str(tmp_path / "trial-0001"),
                    "command": "",
                    "workflow_path": "",
                    "metrics_path": str(tmp_path / "trial-0001" / "metrics.json"),
                },
            ],
        }
    )
    for trial, discounted_returns in zip(
        compilation.trials,
        ([1.0, 2.0, 3.0], [1.0, 5.0, 7.0]),
        strict=True,
    ):
        history_path = Path(trial.run_dir) / "logs" / "train_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            "\n".join(
                (
                    f'{{"episode": {idx}, "discounted_return": {discounted_return}, '
                    '"return": 0.0, "length": 1, "loss": 0.0}'
                )
                for idx, discounted_return in enumerate(discounted_returns)
            ),
            encoding="utf-8",
        )
    (tmp_path / "sweep_manifest.yaml").write_text(compilation.model_dump_json(), encoding="utf-8")

    summary = summarize_groups(
        tmp_path / "sweep_manifest.yaml",
        metric="mean_train_discounted_return_last_n",
        goal="maximize",
        metric_last_n=2,
    )

    assert summary.iloc[0]["metric_mean"] == 4.25


def test_filter_top_k_curves_keeps_expected_groups() -> None:
    curves = pd.DataFrame(
        [
            {"group_key": "group-0000", "x": 0.0, "mean": 1.0, "ci_low": 1.0, "ci_high": 1.0, "parameters": {"lr": 0.1}},
            {"group_key": "group-0001", "x": 0.0, "mean": 2.0, "ci_low": 2.0, "ci_high": 2.0, "parameters": {"lr": 0.2}},
        ]
    )
    summary = pd.DataFrame(
        [
            {"group_key": "group-0001", "metric_mean": 2.0},
            {"group_key": "group-0000", "metric_mean": 1.0},
        ]
    )
    filtered = filter_top_k_curves(curves, summary, top_k=1)
    assert set(filtered["group_key"]) == {"group-0001"}


def test_export_summary_tables_writes_all_formats(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "rank": 1,
                "group_key": "{}",
                "parameters": {"lr": 0.1},
                "metric_mean": 2.0,
                "metric_min": 2.0,
                "metric_max": 2.0,
                "metric_std": 0.0,
                "seed_count": 2,
                "trial_ids": ["trial-0000", "trial-0001"],
                "run_dirs": ["/tmp/trial-0000", "/tmp/trial-0001"],
            }
        ]
    )
    paths = export_summary_tables(summary, out_dir=tmp_path)
    assert (Path(paths["csv"]).exists())
    assert (Path(paths["json"]).exists())
    assert (Path(paths["markdown"]).exists())


def test_export_summary_tables_rejects_empty_summary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No summary data"):
        export_summary_tables(pd.DataFrame(), out_dir=tmp_path)
