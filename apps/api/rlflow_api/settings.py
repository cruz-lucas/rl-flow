from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RLFLOW_", env_file=".env")

    run_root: Path = Path("runs")
    db_path: Path = Path("runs/rlflow.db")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )


def get_settings() -> Settings:
    return Settings()
