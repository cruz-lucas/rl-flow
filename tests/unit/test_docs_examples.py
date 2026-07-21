"""Guard against documentation drifting away from the configs it references.

Every ``configs/**/*.yaml`` path mentioned in the README or the docs must point at
a file that actually exists, so a reader following the docs never hits a
``file not found`` on their first command.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_REF = re.compile(r"configs/[A-Za-z0-9_./-]+\.yaml")


def _doc_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    files.extend(sorted((REPO_ROOT / "docs").rglob("*.md")))
    return [path for path in files if path.exists()]


def test_docs_reference_only_existing_configs() -> None:
    missing: list[str] = []
    for doc in _doc_files():
        text = doc.read_text(encoding="utf-8")
        for ref in sorted(set(CONFIG_REF.findall(text))):
            if not (REPO_ROOT / ref).exists():
                missing.append(f"{doc.relative_to(REPO_ROOT)} -> {ref}")

    assert not missing, "Docs reference configs that do not exist:\n" + "\n".join(missing)
