from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rlflow.execution.base import ExecutionBackend
from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo, JobState, JobStatus
from rlflow.schemas.resources import SlurmOptions
from rlflow.schemas.sweep import SweepCompilation
from rlflow.tracking.status import RunStatusState, update_status


class SlurmExecutor(ExecutionBackend):
    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}

    @staticmethod
    def render_script(experiment: ExperimentSpec, options: dict[str, Any] | SlurmOptions | None = None) -> str:
        slurm_options = options if isinstance(options, SlurmOptions) else SlurmOptions(**(options or {}))
        template_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=()),
            keep_trailing_newline=True,
        )
        template = env.get_template("slurm_job.sh.j2")
        return template.render(experiment=experiment, options=slurm_options)

    @staticmethod
    def render_array_script(
        sweep: SweepCompilation,
        options: dict[str, Any] | SlurmOptions | None = None,
        *,
        max_parallel: int | None = None,
    ) -> str:
        if not sweep.trials:
            raise ValueError("Cannot render a SLURM array script for an empty sweep")
        slurm_options = options if isinstance(options, SlurmOptions) else SlurmOptions(**(options or {}))
        template_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=()),
            keep_trailing_newline=True,
        )
        template = env.get_template("slurm_array.sh.j2")
        tasks = [
            {
                "command": shlex.quote(trial.command),
                "run_dir": shlex.quote(trial.run_dir),
            }
            for trial in sweep.trials
        ]
        return template.render(
            sweep=sweep,
            options=slurm_options,
            tasks=tasks,
            last_index=len(sweep.trials) - 1,
            max_parallel=max_parallel,
        )

    def submit(self, experiment: ExperimentSpec) -> JobInfo:
        run_dir = Path(experiment.run_dir)
        script_path = run_dir / "slurm_job.sh"
        if not script_path.exists():
            script_path.write_text(
                self.render_script(experiment, experiment.workflow.execution.options),
                encoding="utf-8",
            )

        if shutil.which("sbatch") is None:
            raise RuntimeError("sbatch was not found on PATH; render succeeded but SLURM is unavailable")

        result = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=run_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        external_id = result.stdout.strip().split()[-1]
        job_id = f"slurm-{external_id}"
        update_status(
            run_dir,
            RunStatusState.queued,
            backend="slurm",
            external_id=external_id,
            message=result.stdout.strip(),
        )
        job = JobInfo(
            job_id=job_id,
            experiment_id=experiment.experiment_id,
            backend="slurm",
            status=JobStatus(state=JobState.pending, message=result.stdout.strip()),
            run_dir=str(run_dir),
            external_id=external_id,
            stdout_path=str(run_dir / "logs" / f"slurm-{external_id}.out"),
            stderr_path=str(run_dir / "logs" / f"slurm-{external_id}.err"),
        )
        self._jobs[job_id] = job
        return job

    def submit_array(self, sweep: SweepCompilation) -> JobInfo:
        sweep_dir = Path(sweep.sweep_dir)
        script_path = Path(sweep.slurm_array_path or sweep_dir / "slurm_array.sh")
        if not script_path.exists():
            raise RuntimeError(f"SLURM array script does not exist: {script_path}")

        if shutil.which("sbatch") is None:
            raise RuntimeError("sbatch was not found on PATH; render succeeded but SLURM is unavailable")

        result = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=sweep_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        external_id = result.stdout.strip().split()[-1]
        job_id = f"slurm-array-{external_id}"
        for index, trial in enumerate(sweep.trials):
            update_status(
                trial.run_dir,
                RunStatusState.queued,
                backend="slurm",
                external_id=f"{external_id}_{index}",
                message=result.stdout.strip(),
            )
        job = JobInfo(
            job_id=job_id,
            experiment_id=sweep.sweep_id,
            backend="slurm",
            status=JobStatus(state=JobState.pending, message=result.stdout.strip()),
            run_dir=str(sweep_dir),
            external_id=external_id,
            stdout_path=str(sweep_dir / "logs" / f"slurm-array-{external_id}_%a.out"),
            stderr_path=str(sweep_dir / "logs" / f"slurm-array-{external_id}_%a.err"),
        )
        self._jobs[job_id] = job
        return job

    def status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id)
        external_id = job.external_id if job else self._external_id_from_job_id(job_id)
        if not external_id:
            return JobStatus(state=JobState.unknown, message="missing SLURM job id")

        if shutil.which("squeue"):
            result = subprocess.run(
                ["squeue", "-h", "-j", external_id, "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
            )
            state = result.stdout.strip().splitlines()
            if state:
                return JobStatus(state=self._map_state(state[0]), message=state[0])

        if shutil.which("sacct"):
            result = subprocess.run(
                ["sacct", "-j", external_id, "--format=State", "--noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            state = result.stdout.strip().splitlines()
            if state:
                clean_state = state[0].strip().split()[0]
                return JobStatus(state=self._map_state(clean_state), message=clean_state)

        return JobStatus(state=JobState.unknown, message="SLURM status commands unavailable or job not found")

    def cancel(self, job_id: str) -> None:
        external_id = self._external_id_from_job_id(job_id)
        job = self._jobs.get(job_id)
        if job and job.external_id:
            external_id = job.external_id
        if shutil.which("scancel") is None:
            raise RuntimeError("scancel was not found on PATH")
        subprocess.run(["scancel", external_id], check=True)

    def logs(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return ""
        chunks = []
        for label, path in (("stdout", job.stdout_path), ("stderr", job.stderr_path)):
            if path and Path(path).exists():
                chunks.append(f"== {label} ==\n{Path(path).read_text(encoding='utf-8', errors='replace')}")
        return "\n".join(chunks)

    def _external_id_from_job_id(self, job_id: str) -> str:
        if job_id.startswith("slurm-array-"):
            return job_id.removeprefix("slurm-array-")
        return job_id.removeprefix("slurm-")

    def _map_state(self, slurm_state: str) -> JobState:
        normalized = slurm_state.upper()
        if normalized in {"PENDING", "CONFIGURING"}:
            return JobState.pending
        if normalized in {"RUNNING", "COMPLETING"}:
            return JobState.running
        if normalized in {"COMPLETED"}:
            return JobState.succeeded
        if normalized in {"CANCELLED", "TIMEOUT"}:
            return JobState.cancelled
        if normalized in {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}:
            return JobState.failed
        return JobState.unknown
