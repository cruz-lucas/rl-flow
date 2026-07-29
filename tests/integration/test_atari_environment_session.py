"""End-to-end proof that the Environment playground can run Montezuma's Revenge.

Gated on ale-py/gymnasium (the ``atari`` extra); skipped in CI where they are not
installed. Drives the real session route: create -> step -> reset -> export PDF.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from fastapi.testclient import TestClient

from rlflow_api.main import create_app


def test_atari_playground_session_steps_resets_and_exports_pdf(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())

    created = client.post(
        "/environment-sessions",
        json={
            "component_id": "builtin.env.atari",
            "config": {"task_id": "MontezumaRevenge-v5"},
            "seed": 0,
        },
    )
    assert created.status_code == 200, created.text
    session = created.json()
    assert session["component_id"] == "builtin.env.atari"
    assert session["action_count"] == 18
    assert len(session["action_labels"]) == 18
    assert session["observation_shape"] == [210, 160, 3]
    assert session["svg"].startswith("<svg")
    assert "data:image/png;base64," in session["svg"]

    stepped = client.post(
        f"/environment-sessions/{session['session_id']}/actions", json={"action": 1}
    )
    assert stepped.status_code == 200
    assert stepped.json()["step"] == 1

    reset = client.post(f"/environment-sessions/{session['session_id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["step"] == 0

    pdf = client.get(f"/environment-sessions/{session['session_id']}/export.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-1.4")


def test_atari_playground_rejects_out_of_range_action(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())
    session = client.post(
        "/environment-sessions",
        json={"component_id": "builtin.env.atari", "config": {}, "seed": 0},
    ).json()
    bad = client.post(
        f"/environment-sessions/{session['session_id']}/actions", json={"action": 999}
    )
    assert bad.status_code == 422
