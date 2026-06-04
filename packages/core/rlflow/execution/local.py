from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from rlflow.execution.base import ExecutionBackend
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo, JobState, JobStatus
from rlflow.tracking.status import RunStatus, RunStatusState, load_status, update_status


class LocalExecutor(ExecutionBackend):
    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._jobs: dict[str, JobInfo] = {}

    def submit(self, experiment: ExperimentSpec) -> JobInfo:
        run_dir = Path(experiment.run_dir)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / "local.out"
        stderr_path = logs_dir / "local.err"
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        update_status(run_dir, RunStatusState.queued, backend="local")
        process = subprocess.Popen(
            ["/usr/bin/env", "bash", str(Path(experiment.command).resolve())],
            cwd=run_dir,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        job_id = f"local-{process.pid}"
        job = JobInfo(
            job_id=job_id,
            experiment_id=experiment.experiment_id,
            backend="local",
            status=JobStatus(state=JobState.running, message=f"pid {process.pid}"),
            run_dir=str(run_dir),
            external_id=str(process.pid),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        update_status(
            run_dir,
            RunStatusState.running,
            backend="local",
            external_id=str(process.pid),
            message=f"pid {process.pid}",
        )
        self._processes[job_id] = process
        self._jobs[job_id] = job
        return job

    def status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id)
        if job is not None:
            return self.status_for_job(job)
        return JobStatus(state=JobState.unknown, message="job is not known to this process")

    def status_for_job(self, job: JobInfo) -> JobStatus:
        run_dir = Path(job.run_dir)
        status = load_status(run_dir)
        job_id = job.job_id
        process = self._processes.get(job_id)
        if status is not None and (
            status.status
            in {
                RunStatusState.completed,
                RunStatusState.failed,
                RunStatusState.cancelled,
                RunStatusState.unknown,
            }
            or process is None
        ):
            mapped = self._job_status_from_run_status(status)
            if mapped is not None:
                return mapped

        if process is not None:
            code = process.poll()
            if code is None:
                return JobStatus(state=JobState.running, message=f"pid {process.pid}")
            if code == 0:
                update_status(
                    run_dir,
                    RunStatusState.completed,
                    backend="local",
                    external_id=str(process.pid),
                    exit_code=code,
                )
                return JobStatus(state=JobState.succeeded, message="process exited with code 0")
            if code < 0:
                update_status(
                    run_dir,
                    RunStatusState.cancelled,
                    backend="local",
                    external_id=str(process.pid),
                    exit_code=code,
                    message=f"process terminated by signal {-code}",
                )
                return JobStatus(state=JobState.cancelled, message=f"process terminated by signal {-code}")
            update_status(
                run_dir,
                RunStatusState.failed,
                backend="local",
                external_id=str(process.pid),
                exit_code=code,
                message=f"process exited with code {code}",
            )
            return JobStatus(state=JobState.failed, message=f"process exited with code {code}")

        if self._metrics_exist(run_dir):
            return JobStatus(state=JobState.succeeded, message="metrics summary exists")

        if job.external_id is None:
            return JobStatus(state=JobState.unknown, message="job has no local pid")
        try:
            os.kill(int(job.external_id), 0)
        except ProcessLookupError:
            return JobStatus(state=JobState.unknown, message="process is no longer running")
        return JobStatus(state=JobState.running, message=f"pid {job.external_id}")

    def cancel(self, job_id: str) -> None:
        process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            return
        job = self._jobs.get(job_id)
        if job and job.external_id:
            try:
                os.kill(int(job.external_id), signal.SIGTERM)
            except ProcessLookupError:
                return

    def logs(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return ""
        chunks = []
        for label, path in (("stdout", job.stdout_path), ("stderr", job.stderr_path)):
            if path and Path(path).exists():
                chunks.append(f"== {label} ==\n{Path(path).read_text(encoding='utf-8', errors='replace')}")
        return "\n".join(chunks)

    def _job_status_from_run_status(self, status: RunStatus) -> JobStatus | None:
        if status.status == RunStatusState.completed:
            return JobStatus(state=JobState.succeeded, message=status.message or "run completed")
        if status.status == RunStatusState.failed:
            return JobStatus(state=JobState.failed, message=status.message or "run failed")
        if status.status == RunStatusState.cancelled:
            return JobStatus(state=JobState.cancelled, message=status.message or "run cancelled")
        if status.status == RunStatusState.running:
            return JobStatus(state=JobState.running, message=status.message or "run is running")
        if status.status == RunStatusState.queued:
            return JobStatus(state=JobState.pending, message=status.message or "run is queued")
        if status.status == RunStatusState.unknown:
            return JobStatus(state=JobState.unknown, message=status.message or "run status is unknown")
        return None

    def _metrics_exist(self, run_dir: Path) -> bool:
        return (
            (run_dir / "summaries" / "metrics.json").exists()
            or (run_dir / "metrics.json").exists()
        )
