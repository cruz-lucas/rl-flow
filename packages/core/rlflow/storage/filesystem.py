from __future__ import annotations

from pathlib import Path


class FilesystemArtifactStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self, run_dir: str | Path) -> list[str]:
        base = Path(run_dir)
        if not base.exists():
            return []
        return [str(path) for path in sorted(base.rglob("*")) if path.is_file()]

    def read_text(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")
