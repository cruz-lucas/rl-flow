import subprocess
from pathlib import Path

from rlflow.execution.slurm import SlurmExecutor
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.sweep import SweepCompilation, SweepMetric, SweepTrial
from rlflow.schemas.workflow import WorkflowSpec
from rlflow.tracking.status import load_status


def _sweep_compilation(tmp_path: Path, trial_count: int, *, trials_per_task: int = 1) -> SweepCompilation:
    trials = [
        SweepTrial(
            index=index,
            trial_id=f"trial-{index:04d}",
            experiment_id=f"trial-{index:04d}",
            parameters={"index": index},
            run_dir=str(tmp_path / "trials" / f"trial-{index:04d}"),
            command=str(tmp_path / "trials" / f"trial-{index:04d}" / "command.sh"),
            workflow_path=str(tmp_path / "trials" / f"trial-{index:04d}" / "workflow.yaml"),
            metrics_path=str(tmp_path / "trials" / f"trial-{index:04d}" / "summaries" / "metrics.json"),
        )
        for index in range(trial_count)
    ]
    return SweepCompilation(
        sweep_id="sweep",
        name="sweep",
        method="grid",
        metric=SweepMetric(),
        sweep_dir=str(tmp_path),
        manifest_path=str(tmp_path / "sweep_manifest.yaml"),
        slurm_array_path=str(tmp_path / "slurm_array.sh"),
        slurm_trials_per_task=trials_per_task,
        slurm_array_task_count=(trial_count + trials_per_task - 1) // trials_per_task,
        trials=trials,
    )


def test_slurm_executor_renders_expected_script(tmp_path) -> None:
    experiment = ExperimentSpec(
        experiment_id="exp",
        workflow=WorkflowSpec(name="wf", nodes=[]),
        resolved_config={},
        run_dir=str(tmp_path),
        command=str(tmp_path / "command.sh"),
        generated_files=[],
        execution_backend="slurm",
    )

    script = SlurmExecutor.render_script(
        experiment,
        {"partition": "gpu", "account": "abc", "time": "00:10:00", "cpus_per_task": 2, "mem": "8G"},
    )

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=abc" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "export RLFLOW_BACKEND=slurm" in script
    assert 'export RLFLOW_EXTERNAL_ID="${SLURM_JOB_ID:-}"' in script
    assert f'bash "{experiment.command}"' in script


def test_slurm_executor_renders_batched_array_script(tmp_path: Path) -> None:
    compilation = _sweep_compilation(tmp_path, 10, trials_per_task=4)

    script = SlurmExecutor.render_array_script(
        compilation,
        {"time": "00:10:00", "cpus_per_task": 1, "mem": "4G"},
        max_parallel=3,
        trials_per_task=4,
    )

    assert "#SBATCH --array=0-2%3" in script
    assert "TRIALS_PER_TASK=4" in script
    assert "TOTAL_TRIALS=10" in script
    assert "START_INDEX=$((TASK_ID * TRIALS_PER_TASK))" in script
    assert "END_INDEX=$((START_INDEX + TRIALS_PER_TASK))" in script
    assert "for ((TRIAL_INDEX = START_INDEX; TRIAL_INDEX < END_INDEX; TRIAL_INDEX++)); do" in script
    assert 'if ! bash "$COMMAND"; then' in script


def test_slurm_executor_submit_array_queues_trials_with_batch_task_ids(tmp_path: Path, monkeypatch) -> None:
    compilation = _sweep_compilation(tmp_path, 5, trials_per_task=2)
    Path(compilation.slurm_array_path or "").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.setattr("rlflow.execution.slurm.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "rlflow.execution.slurm.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="Submitted batch job 12345\n"),
    )

    job = SlurmExecutor().submit_array(compilation)

    assert job.external_id == "12345"
    assert [load_status(trial.run_dir).external_id for trial in compilation.trials] == [
        "12345_0",
        "12345_0",
        "12345_1",
        "12345_1",
        "12345_2",
    ]
