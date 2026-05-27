from pathlib import Path

import yaml

from rlflow.graph.sweep import SweepCompiler
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.sweep import SweepSpec


def test_sweep_compiler_writes_trial_workflows_and_slurm_array(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(
        {
            "name": "navix sweep",
            "sweep_id": "test-sweep",
            "workflow": "configs/workflows/navix_dqn_empty_room.yaml",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "execution": {
                "backend": "slurm",
                "options": {
                    "account": "def-example",
                    "time": "00:10:00",
                    "cpus_per_task": 2,
                    "mem": "8G",
                    "gres": "gpu:1",
                },
            },
            "slurm": {"max_parallel": 2},
            "parameters": {
                "lr": {
                    "target": "nodes.agent.config.learning_rate",
                    "values": [0.001, 0.0003],
                },
                "batch": {
                    "target": "nodes.replay.config.batch_size",
                    "values": [32, 64],
                },
            },
        }
    )

    compilation = SweepCompiler(create_default_registry(discover=False)).compile(spec, out_dir=tmp_path)

    assert len(compilation.trials) == 4
    assert Path(compilation.manifest_path).exists()
    assert compilation.slurm_array_path is not None
    script = Path(compilation.slurm_array_path).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3%2" in script
    assert "#SBATCH --account=def-example" in script
    assert "#SBATCH --gres=gpu:1" in script

    workflow = yaml.safe_load(Path(compilation.trials[0].workflow_path).read_text(encoding="utf-8"))
    agent = next(node for node in workflow["nodes"] if node["id"] == "agent")
    replay = next(node for node in workflow["nodes"] if node["id"] == "replay")
    assert agent["config"]["learning_rate"] == 0.001
    assert replay["config"]["batch_size"] == 32
    assert workflow["metadata"]["experiment_id"] == "test-sweep-trial-0000"
    assert workflow["metadata"]["sweep_parameters"] == {"batch": 32, "lr": 0.001}


def test_sweep_summarize_selects_best_metric(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(
        {
            "name": "summary sweep",
            "sweep_id": "summary-sweep",
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
    compiler = SweepCompiler(create_default_registry(discover=False))
    compilation = compiler.compile(spec, out_dir=tmp_path)
    Path(compilation.trials[0].metrics_path).write_text('{"mean_eval_return": 1.0}', encoding="utf-8")
    Path(compilation.trials[1].metrics_path).write_text('{"mean_eval_return": 3.0}', encoding="utf-8")

    summary = compiler.summarize(compilation.manifest_path)

    assert summary["best"]["parameters"] == {}
    assert summary["best"]["metric"] == 2.0
    assert summary["best"]["metric_count"] == 2


def test_sweep_summarize_computes_train_return_last_n_from_history(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(
        {
            "name": "history sweep",
            "sweep_id": "history-sweep",
            "workflow": "configs/workflows/tabular_q_learning_riverswim.yaml",
            "method": "grid",
            "metric": {"name": "mean_train_return_last_n", "goal": "maximize", "last_n": 2},
            "parameters": {
                "seed": {
                    "target": "nodes.runner.config.seed",
                    "values": [0, 1],
                },
            },
        }
    )
    compiler = SweepCompiler(create_default_registry(discover=False))
    compilation = compiler.compile(spec, out_dir=tmp_path)
    histories = [
        [1.0, 2.0, 3.0],
        [1.0, 5.0, 7.0],
    ]
    for trial, returns in zip(compilation.trials, histories, strict=True):
        history_path = Path(trial.run_dir) / "logs" / "train_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            "\n".join(
                f'{{"episode": {idx}, "return": {episode_return}, "length": 1, "loss": 0.0}}'
                for idx, episode_return in enumerate(returns)
            ),
            encoding="utf-8",
        )

    summary = compiler.summarize(compilation.manifest_path, metric_last_n=2)

    assert summary["metric_last_n"] == 2
    assert summary["best"]["parameters"] == {}
    assert summary["best"]["metric"] == 4.25
    assert summary["best"]["metric_count"] == 2


def test_sweep_summarize_averages_seed_replicates_by_configuration(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(
        {
            "name": "replicate sweep",
            "sweep_id": "replicate-sweep",
            "workflow": "configs/workflows/tabular_q_learning_riverswim.yaml",
            "method": "grid",
            "metric": {"name": "mean_eval_return", "goal": "maximize"},
            "parameters": {
                "lr": {
                    "target": "nodes.agent.config.learning_rate",
                    "values": [0.01, 0.1],
                },
                "seed": {
                    "target": "nodes.runner.config.seed",
                    "values": [0, 1],
                },
            },
        }
    )
    compiler = SweepCompiler(create_default_registry(discover=False))
    compilation = compiler.compile(spec, out_dir=tmp_path)
    metrics = [10.0, 0.0, 4.0, 4.0]
    for trial, metric in zip(compilation.trials, metrics, strict=True):
        Path(trial.metrics_path).write_text(f'{{"mean_eval_return": {metric}}}', encoding="utf-8")

    summary = compiler.summarize(compilation.manifest_path)

    assert summary["best"]["parameters"] == {"lr": 0.01}
    assert summary["best"]["metric"] == 5.0
    assert summary["best"]["metric_count"] == 2
    assert summary["groups"][1]["parameters"] == {"lr": 0.1}
    assert summary["groups"][1]["metric"] == 4.0
