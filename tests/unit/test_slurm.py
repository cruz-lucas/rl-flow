from rlflow.execution.slurm import SlurmExecutor
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.workflow import WorkflowSpec


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
