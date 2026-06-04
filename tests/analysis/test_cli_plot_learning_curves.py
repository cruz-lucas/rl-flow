from pathlib import Path

import yaml
from typer.testing import CliRunner

from rlflow.cli import app


def test_plot_learning_curves_uses_config_labels_and_outputs(tmp_path: Path) -> None:
    sweep_dir = _write_sweep(tmp_path / "sweep")
    config_path = tmp_path / "plot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "figure": {
                    "width": 3.0,
                    "height": 2.0,
                    "dpi": 150,
                    "x_label": "Environment steps",
                    "y_label": "Return",
                    "legend_title": "Agent",
                },
                "curves": {
                    "history": "train",
                    "x": "env_step",
                    "y": "return",
                    "points": 3,
                    "bootstrap_samples": 0,
                    "groups": ["group-0001"],
                    "smooth_window": 2,
                },
                "labels": {'{"agent": "rnd"}': "RND"},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "plots"

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            "plot-learning-curves",
            str(sweep_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "learning_curve.pdf").exists()
    assert (out_dir / "learning_curve.svg").exists()
    assert (out_dir / "learning_curve.png").exists()
    assert (out_dir / "curves_raw.csv").exists()
    assert (out_dir / "curves_interpolated.csv").exists()
    assert (out_dir / "curves_plot.csv").exists()
    assert (out_dir / "plot_config.yaml").exists()

    curves_csv = (out_dir / "curves_interpolated.csv").read_text(encoding="utf-8")
    assert "group-0001" in curves_csv
    assert "group-0000" not in curves_csv
    assert "RND" in curves_csv

    resolved_config = yaml.safe_load((out_dir / "plot_config.yaml").read_text(encoding="utf-8"))
    assert resolved_config["curves"]["groups"] == ["group-0001"]
    assert resolved_config["curves"]["smooth_window"] == 2


def test_plot_learning_curves_accepts_groups_after_single_option(tmp_path: Path) -> None:
    sweep_dir = _write_sweep(tmp_path / "sweep")
    out_dir = tmp_path / "plots-cli"

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            "plot-learning-curves",
            str(sweep_dir),
            "--y",
            "return",
            "--points",
            "3",
            "--bootstrap-samples",
            "0",
            "--out",
            str(out_dir),
            "--groups",
            "group-0000",
            "group-0001",
        ],
    )

    assert result.exit_code == 0, result.output
    resolved_config = yaml.safe_load((out_dir / "plot_config.yaml").read_text(encoding="utf-8"))
    assert resolved_config["curves"]["groups"] == ["group-0000", "group-0001"]


def _write_sweep(sweep_dir: Path) -> Path:
    trials = []
    for index, (group_id, agent, seed, returns) in enumerate(
        [
            ("group-0000", "dqn", 0, [1.0, 2.0]),
            ("group-0000", "dqn", 1, [1.5, 2.5]),
            ("group-0001", "rnd", 0, [3.0, 4.0]),
            ("group-0001", "rnd", 1, [3.5, 4.5]),
        ]
    ):
        trial_id = f"trial-{index:04d}"
        run_dir = sweep_dir / "trials" / group_id / f"seed-{seed}"
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "train_history.jsonl").write_text(
            "\n".join(
                [
                    (
                        '{"episode": %d, "env_step": %d, "return": %.1f, '
                        '"discounted_return": %.1f, "length": 1, "loss": 0.0}'
                    )
                    % (episode, episode + 1, value, value)
                    for episode, value in enumerate(returns)
                ]
            ),
            encoding="utf-8",
        )
        trials.append(
            {
                "index": index,
                "trial_id": trial_id,
                "group_id": group_id,
                "group_run_dir": str(run_dir.parent),
                "seed_value": seed,
                "experiment_id": f"exp-{index}",
                "parameters": {"agent": agent, "seed": seed},
                "run_dir": str(run_dir),
                "command": "",
                "workflow_path": str(run_dir / "workflow.yaml"),
                "metrics_path": str(run_dir / "metrics.json"),
            }
        )

    manifest = {
        "sweep_id": "plot-sweep",
        "name": "plot sweep",
        "method": "grid",
        "metric": {"name": "mean_train_return_last_10", "goal": "maximize"},
        "sweep_dir": str(sweep_dir),
        "manifest_path": str(sweep_dir / "sweep_manifest.yaml"),
        "slurm_array_path": None,
        "generated_files": [],
        "trials": trials,
    }
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return sweep_dir
