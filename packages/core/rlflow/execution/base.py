from __future__ import annotations

from abc import ABC, abstractmethod

from rlflow.schemas.experiment import ExperimentSpec
from rlflow.schemas.job import JobInfo, JobStatus


class ExecutionBackend(ABC):
    @abstractmethod
    def submit(self, experiment: ExperimentSpec) -> JobInfo:
        raise NotImplementedError

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def logs(self, job_id: str) -> str:
        raise NotImplementedError
