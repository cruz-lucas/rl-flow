from __future__ import annotations

from pydantic import BaseModel, Field


class SlurmOptions(BaseModel):
    partition: str | None = None
    account: str | None = None
    time: str = "01:00:00"
    nodes: int | None = None
    ntasks: int | None = None
    cpus_per_task: int = 1
    mem: str = "4G"
    gres: str | None = None
    constraint: str | None = None
    qos: str | None = None
    reservation: str | None = None
    mail_user: str | None = None
    mail_type: str | None = None
    modules: list[str] = Field(default_factory=list)
    venv_path: str | None = None
    conda_env: str | None = None
    preamble: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
