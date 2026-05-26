from __future__ import annotations

from pathlib import Path


def read_logs(*paths: str | None) -> str:
    chunks = []
    for path in paths:
        if path and Path(path).exists():
            chunks.append(Path(path).read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)
