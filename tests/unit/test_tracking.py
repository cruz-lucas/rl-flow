import json
from pathlib import Path

from rlflow.tracking.logger import JsonlLogger
from rlflow.tracking.manifest import RunManifest, file_sha256, load_manifest, write_manifest
from rlflow.tracking.status import RunStatusState, load_status, update_status


def test_manifest_writer_and_sha256(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("hello", encoding="utf-8")

    manifest = RunManifest(
        run_id="run-1",
        experiment_id="run-1",
        created_at="2026-06-03T00:00:00Z",
        run_dir=str(tmp_path),
        workflow_path=str(payload),
        resolved_config_path=str(payload),
        generated_gin_path=str(payload),
        command_path=str(payload),
        workflow_sha256=file_sha256(payload),
        resolved_config_sha256=file_sha256(payload),
        generated_gin_sha256=file_sha256(payload),
        command_sha256=file_sha256(payload),
        python_version="3.11",
        platform="test",
        backend="local",
    )

    write_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path)

    assert loaded.schema_version == "rlflow.run.v1"
    assert loaded.workflow_sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_status_update_preserves_lifecycle_timestamps(tmp_path: Path) -> None:
    compiled = update_status(tmp_path, RunStatusState.compiled, backend="local")
    running = update_status(tmp_path, RunStatusState.running, backend="local", external_id="123")
    completed = update_status(tmp_path, RunStatusState.completed, exit_code=0)

    loaded = load_status(tmp_path)

    assert compiled.created_at == running.created_at == completed.created_at
    assert running.started_at is not None
    assert completed.finished_at is not None
    assert loaded is not None
    assert loaded.status == RunStatusState.completed
    assert loaded.external_id == "123"
    assert loaded.exit_code == 0


def test_jsonl_logger_appends_metrics_and_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("data", encoding="utf-8")

    with JsonlLogger(tmp_path) as logger:
        logger.log_metric("return", 1.5, step=10, phase="summary")
        logger.log_artifact(artifact, name="artifact")

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["type"] == "metric"
    assert rows[0]["name"] == "return"
    assert rows[0]["value"] == 1.5
    assert rows[1]["type"] == "artifact"
    assert rows[1]["path"] == "artifact.txt"
