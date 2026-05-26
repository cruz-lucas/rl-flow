from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def slugify_run_name(name: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    return slug or "run"


def make_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify_run_name(name)}-{timestamp}-{uuid4().hex[:8]}"


def make_flow_run_dir(root: str | Path, flow_name: str, run_id: str) -> Path:
    return Path(root) / slugify_run_name(flow_name) / run_id
