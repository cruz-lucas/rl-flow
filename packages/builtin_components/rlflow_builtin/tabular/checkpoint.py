from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlflow_builtin.tabular.types import TabularRunResult


def save_checkpoint(result: TabularRunResult, checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_path,
        q_table=result.q_table,
        action_counts=result.action_counts,
        metadata=np.asarray([metadata], dtype=object),
    )
