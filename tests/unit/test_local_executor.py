import time
from pathlib import Path

from rlflow.execution.local import LocalExecutor
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.workflow import WorkflowSpec


def test_local_executor_creates_process_and_logs(tmp_path: Path) -> None:
    command = tmp_path / "command.sh"
    command.write_text("#!/usr/bin/env bash\necho hello\n", encoding="utf-8")
    experiment = ExperimentSpec(
        experiment_id="exp",
        workflow=WorkflowSpec(name="wf", nodes=[]),
        resolved_config={},
        run_dir=str(tmp_path),
        command=str(command),
        generated_files=[str(command)],
        execution_backend="local",
    )
    executor = LocalExecutor()

    job = executor.submit(experiment)
    for _ in range(20):
        status = executor.status(job.job_id)
        if status.state != "running":
            break
        time.sleep(0.05)

    assert (tmp_path / "logs" / "local.out").exists()
    assert "hello" in (tmp_path / "logs" / "local.out").read_text(encoding="utf-8")
