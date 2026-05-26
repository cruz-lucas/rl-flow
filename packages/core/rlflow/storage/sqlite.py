from __future__ import annotations

import re
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo, JobState, JobStatus
from rlflow.schemas.workflow import WorkflowSpec
from rlflow.storage.models import ExperimentRecord, JobRecord, WorkflowRecord, utc_now


def _workflow_id_from_name(name: str) -> str:
    workflow_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("._-").lower()
    return workflow_id or "untitled_workflow"


class Storage:
    def __init__(self, url: str = "sqlite:///./rlflow.db") -> None:
        self.engine = create_engine(url, connect_args={"check_same_thread": False})

    @classmethod
    def from_path(cls, path: str | Path) -> "Storage":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{path}")

    def init(self) -> None:
        SQLModel.metadata.create_all(self.engine)

    def save_experiment(self, experiment: ExperimentSpec, status: str = "compiled") -> ExperimentRecord:
        with Session(self.engine) as session:
            record = session.get(ExperimentRecord, experiment.experiment_id)
            if record is None:
                record = ExperimentRecord(
                    experiment_id=experiment.experiment_id,
                    workflow_spec=experiment.workflow.model_dump(mode="json"),
                    resolved_config=experiment.resolved_config,
                    run_dir=experiment.run_dir,
                    command=experiment.command,
                    generated_files=experiment.generated_files,
                    execution_backend=experiment.execution_backend,
                    status=status,
                )
                session.add(record)
            else:
                record.workflow_spec = experiment.workflow.model_dump(mode="json")
                record.resolved_config = experiment.resolved_config
                record.run_dir = experiment.run_dir
                record.command = experiment.command
                record.generated_files = experiment.generated_files
                record.execution_backend = experiment.execution_backend
                record.status = status
                record.updated_at = utc_now()
            session.commit()
            session.refresh(record)
            return record

    def list_experiments(self) -> list[ExperimentRecord]:
        with Session(self.engine) as session:
            return list(session.exec(select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())).all())

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with Session(self.engine) as session:
            return session.get(ExperimentRecord, experiment_id)

    def update_experiment_status(self, experiment_id: str, status: str) -> ExperimentRecord | None:
        with Session(self.engine) as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                return None
            record.status = status
            record.updated_at = utc_now()
            session.commit()
            session.refresh(record)
            return record

    def save_workflow(self, workflow: WorkflowSpec, workflow_id: str | None = None) -> WorkflowRecord:
        workflow_id = workflow_id or _workflow_id_from_name(workflow.name)
        with Session(self.engine) as session:
            record = session.get(WorkflowRecord, workflow_id)
            if record is None:
                record = WorkflowRecord(
                    workflow_id=workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    workflow_spec=workflow.model_dump(mode="json"),
                )
                session.add(record)
            else:
                record.name = workflow.name
                record.description = workflow.description
                record.workflow_spec = workflow.model_dump(mode="json")
                record.updated_at = utc_now()
            session.commit()
            session.refresh(record)
            return record

    def list_workflows(self) -> list[WorkflowRecord]:
        with Session(self.engine) as session:
            return list(session.exec(select(WorkflowRecord).order_by(WorkflowRecord.updated_at.desc())).all())

    def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        with Session(self.engine) as session:
            return session.get(WorkflowRecord, workflow_id)

    def save_job(self, job: JobInfo) -> JobRecord:
        with Session(self.engine) as session:
            record = session.get(JobRecord, job.job_id)
            if record is None:
                record = JobRecord(
                    job_id=job.job_id,
                    experiment_id=job.experiment_id,
                    backend=job.backend,
                    status=job.status.state.value,
                    message=job.status.message,
                    run_dir=job.run_dir,
                    external_id=job.external_id,
                    stdout_path=job.stdout_path,
                    stderr_path=job.stderr_path,
                )
                session.add(record)
            else:
                record.status = job.status.state.value
                record.message = job.status.message
                record.external_id = job.external_id
                record.stdout_path = job.stdout_path
                record.stderr_path = job.stderr_path
                record.updated_at = utc_now()
            session.commit()
            session.refresh(record)
            return record

    def update_job_status(self, job_id: str, status: JobStatus) -> JobRecord | None:
        with Session(self.engine) as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                return None
            record.status = status.state.value
            record.message = status.message
            record.updated_at = utc_now()
            session.commit()
            session.refresh(record)
            return record

    def list_jobs(self) -> list[JobRecord]:
        with Session(self.engine) as session:
            return list(session.exec(select(JobRecord).order_by(JobRecord.created_at.desc())).all())

    def get_job(self, job_id: str) -> JobRecord | None:
        with Session(self.engine) as session:
            return session.get(JobRecord, job_id)

    def job_info_from_record(self, record: JobRecord) -> JobInfo:
        return JobInfo(
            job_id=record.job_id,
            experiment_id=record.experiment_id,
            backend=record.backend,
            status=JobStatus(state=JobState(record.status), message=record.message),
            run_dir=record.run_dir,
            external_id=record.external_id,
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
        )
