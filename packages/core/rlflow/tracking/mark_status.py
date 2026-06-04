from __future__ import annotations

import argparse
import os
from pathlib import Path

from rlflow.tracking.status import RunStatusState, update_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Update an rl-flow run status.json file")
    parser.add_argument("state", choices=[state.value for state in RunStatusState])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--message")
    parser.add_argument("--backend")
    parser.add_argument("--external-id")
    args = parser.parse_args()

    update_status(
        Path(args.run_dir),
        args.state,
        exit_code=args.exit_code,
        message=args.message,
        backend=args.backend or os.environ.get("RLFLOW_BACKEND"),
        external_id=args.external_id or _external_id_from_env(),
    )
    return 0


def _external_id_from_env() -> str | None:
    if os.environ.get("RLFLOW_EXTERNAL_ID"):
        return os.environ["RLFLOW_EXTERNAL_ID"]
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job_id and array_task_id:
        return f"{array_job_id}_{array_task_id}"
    return os.environ.get("SLURM_JOB_ID")


if __name__ == "__main__":
    raise SystemExit(main())
