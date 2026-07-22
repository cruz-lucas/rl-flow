import json
from pathlib import Path

import pytest

from rlflow.analysis.run_report import format_run_report, plot_run_curve, summarize_run


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "summaries").mkdir()
    (run / "logs" / "train_history.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode": 0,
                        "env_step": 10,
                        "return": 1.0,
                        "discounted_return": 0.9,
                        "length": 10,
                        "loss": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "episode": 1,
                        "env_step": 20,
                        "return": 3.0,
                        "discounted_return": 2.4,
                        "length": 10,
                        "loss": 0.3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "logs" / "eval_history.jsonl").write_text(
        json.dumps({"episode": 0, "env_step": 10, "return": 2.0, "discounted_return": 1.8}) + "\n",
        encoding="utf-8",
    )
    (run / "summaries" / "metrics.json").write_text(
        json.dumps({"mean_train_return": 2.0, "agent": "builtin.agent.q_learning_tabular"}),
        encoding="utf-8",
    )
    (run / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "backend": "local",
                "seed": 0,
                "git_commit": "abc123",
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_summarize_run_reports_returns_and_metadata(tmp_path: Path) -> None:
    summary = summarize_run(_make_run(tmp_path))
    assert summary["status"] == "completed"
    assert summary["run_id"] == "r1"
    assert summary["train_episodes"] == 2
    assert summary["eval_episodes"] == 1
    assert summary["total_env_steps"] == 20
    assert summary["train_return"]["final"] == 3.0
    assert summary["train_discounted_return"]["max"] == 2.4
    assert summary["eval_return"]["mean"] == 2.0

    report = format_run_report(summary)
    assert "completed" in report
    assert "discounted_return" in report
    assert "mean_train_return" in report


def test_summarize_run_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_run(tmp_path / "does-not-exist")


def test_plot_run_curve_writes_png(tmp_path: Path) -> None:
    path = plot_run_curve(_make_run(tmp_path), tmp_path / "plots")
    assert path is not None
    assert path.exists()
    assert path.suffix == ".png"
