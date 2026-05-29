from pathlib import Path
import time

from fastapi.testclient import TestClient
import yaml

from rlflow_api.main import create_app


def test_api_health_and_components(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    components = client.get("/components").json()
    assert any(component["id"] == "builtin.agent.dqn_jax" for component in components)
    assert any(component["id"] == "builtin.agent.q_learning_tabular" for component in components)
    assert any(component["id"] == "navix.env.grid" for component in components)
    assert any(component["id"] == "builtin.env.riverswim" for component in components)
    assert any(component["id"] == "builtin.env.sixarms" for component in components)
    assert any(component["id"] == "builtin.replay.tabular_uniform" for component in components)
    assert any(component["source"] == "builtin" for component in components)
    cors_response = client.get("/components", headers={"Origin": "http://127.0.0.1:5173"})
    assert cors_response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_api_saves_and_loads_workflow_gallery(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())
    workflow = yaml.safe_load(Path("configs/workflows/tabular_q_learning_riverswim.yaml").read_text(encoding="utf-8"))

    saved = client.post("/workflows", json={"workflow": workflow}).json()
    assert saved["workflow_id"] == "tabular_q_learning_riverswim"
    assert saved["name"] == "tabular_q_learning_riverswim"

    gallery = client.get("/workflows").json()
    assert [item["workflow_id"] for item in gallery] == ["tabular_q_learning_riverswim"]

    loaded = client.get("/workflows/tabular_q_learning_riverswim").json()
    assert loaded["name"] == workflow["name"]
    assert loaded["nodes"][0]["component"] == "builtin.env.riverswim"


def test_api_compiles_sweep_from_saved_workflow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())
    workflow = yaml.safe_load(Path("configs/workflows/tabular_q_learning_riverswim.yaml").read_text(encoding="utf-8"))
    saved = client.post("/workflows", json={"workflow": workflow}).json()

    candidates = client.get(f"/sweeps/workflows/{saved['workflow_id']}/candidates")

    assert candidates.status_code == 200
    candidate_data = candidates.json()
    assert any(item["target"] == "nodes.agent.config.learning_rate" for item in candidate_data["candidates"])
    seed_target = candidate_data["seed_candidates"][0]["target"]

    compiled = client.post(
        "/sweeps/compile",
        json={
            "workflow_id": saved["workflow_id"],
            "name": "api sweep",
            "method": "grid",
            "metric_name": "mean_eval_return",
            "metric_goal": "maximize",
            "parameters": [
                {
                    "label": "lr",
                    "target": "nodes.agent.config.learning_rate",
                    "values": [0.05, 0.1],
                }
            ],
            "seed_target": seed_target,
            "seed_count": 2,
            "seed_start": 0,
        },
    )

    assert compiled.status_code == 200
    sweep = compiled.json()
    assert len(sweep["trials"]) == 4
    assert Path(sweep["manifest_path"]).exists()
    assert Path(sweep["trials"][0]["workflow_path"]).exists()
    history_path = Path(sweep["trials"][0]["run_dir"]) / "logs" / "train_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\n".join(
            [
                '{"episode": 0, "return": 1.0, "length": 1, "loss": 0.0}',
                '{"episode": 1, "return": 3.0, "length": 1, "loss": 0.0}',
            ]
        ),
        encoding="utf-8",
    )

    listed = client.get("/sweeps")
    assert listed.status_code == 200
    assert listed.json()[0]["sweep_id"] == sweep["sweep_id"]

    inspected = client.post(
        "/sweeps/inspect",
        json={
            "path": sweep["manifest_path"],
            "metric_name": "mean_train_return_last_n",
            "metric_goal": "maximize",
            "metric_last_n": 2,
        },
    )
    assert inspected.status_code == 200
    summary = inspected.json()
    assert summary["best"]["parameters"] == {"lr": 0.05}
    assert summary["best"]["metric"] == 2.0
    assert summary["best"]["metric_count"] == 1


def test_api_run_creates_unique_runs_and_refreshes_job_status(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())
    workflow = yaml.safe_load(Path("configs/workflows/tabular_q_learning_riverswim.yaml").read_text(encoding="utf-8"))
    for node in workflow["nodes"]:
        if node["id"] == "runner":
            node["config"]["train_episodes"] = 2
            node["config"]["max_episode_steps"] = 2
            node["config"]["eval_episodes"] = 0
            node["config"]["save_final_checkpoint"] = False

    first_job = client.post("/experiments/run", json={"workflow": workflow, "backend": "local"}).json()
    second_job = client.post("/experiments/run", json={"workflow": workflow, "backend": "local"}).json()

    assert first_job["experiment_id"] != second_job["experiment_id"]
    assert first_job["run_dir"] != second_job["run_dir"]
    assert Path(first_job["run_dir"]).parent.name == "tabular-q-learning-riverswim"
    assert Path(second_job["run_dir"]).parent.name == "tabular-q-learning-riverswim"

    deadline = time.time() + 10
    terminal_states = {"succeeded", "failed", "cancelled", "unknown"}
    states = {}
    while time.time() < deadline:
        jobs = client.get("/jobs").json()
        states = {job["job_id"]: job["status"]["state"] for job in jobs}
        if (
            states.get(first_job["job_id"]) in terminal_states
            and states.get(second_job["job_id"]) in terminal_states
        ):
            break
        time.sleep(0.1)

    assert states[first_job["job_id"]] == "succeeded"
    assert states[second_job["job_id"]] == "succeeded"

    results = client.get("/experiments/results")
    assert results.status_code == 200
    result_rows = results.json()
    first_result = next(row for row in result_rows if row["experiment_id"] == first_job["experiment_id"])
    assert first_result["metrics"]["mean_train_return"] is not None
    assert first_result["train_history"][0]["episode"] == 0


def test_api_environment_session_steps_and_exports_pdf(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    client = TestClient(create_app())

    created = client.post(
        "/environment-sessions",
        json={
            "component_id": "navix.env.grid",
            "config": {
                "env_name": "empty_room",
                "size": 5,
                "layout": "fixed",
                "observation_mode": "tabular",
                "action_set": "cardinal",
                "max_steps": 20,
            },
            "seed": 0,
        },
    )
    assert created.status_code == 200
    session = created.json()
    assert session["action_labels"] == ["Up", "Down", "Left", "Right"]
    assert session["observation_dtype"] == "int32"
    assert session["observation_preview"] == 0
    assert session["svg"].startswith("<svg")

    stepped = client.post(f"/environment-sessions/{session['session_id']}/actions", json={"action": 3})
    assert stepped.status_code == 200
    assert stepped.json()["step"] == 1

    pdf = client.get(f"/environment-sessions/{session['session_id']}/export.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-1.4")


def test_api_dataset_inspection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(tmp_path / "runs"))
    dataset_path = tmp_path / "replay.npz"
    import numpy as np

    np.savez_compressed(
        dataset_path,
        observations=np.asarray([0, 1], dtype=np.int32),
        actions=np.asarray([1, 0], dtype=np.int32),
        rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        next_observations=np.asarray([1, 2], dtype=np.int32),
        terminals=np.asarray([False, True], dtype=np.bool_),
    )
    client = TestClient(create_app())

    response = client.post("/datasets/inspect", json={"path": str(dataset_path), "preview_rows": 1})

    assert response.status_code == 200
    inspection = response.json()
    assert inspection["is_transition_dataset"] is True
    assert inspection["num_transitions"] == 2
    assert inspection["preview"] == [
        {
            "index": 0,
            "observation": 0,
            "action": 1,
            "reward": 0.0,
            "next_observation": 1,
            "terminal": False,
        }
    ]


def test_api_dataset_resolution_and_visitation_for_navix_symbolic(monkeypatch, tmp_path) -> None:
    run_root = tmp_path / "runs"
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(run_root))
    dataset_path = run_root / "sample" / "replay.npz"
    dataset_path.parent.mkdir(parents=True)
    import numpy as np

    observations = np.stack(
        [
            _symbolic_navix_observation(1, 1),
            _symbolic_navix_observation(1, 2),
            _symbolic_navix_observation(1, 1),
        ]
    )
    np.savez_compressed(
        dataset_path,
        observations=observations,
        actions=np.asarray([0, 3, 1], dtype=np.int32),
        rewards=np.zeros((3,), dtype=np.float32),
        next_observations=observations,
        terminals=np.asarray([False, False, True], dtype=np.bool_),
    )
    client = TestClient(create_app())

    response = client.post("/datasets/inspect", json={"path": "runs/sample/replay", "preview_rows": 1})

    assert response.status_code == 200
    inspection = response.json()
    assert inspection["path"] == str(dataset_path)
    assert inspection["visitation"]["source"] == "navix_symbolic"
    assert inspection["visitation"]["state_counts"][1][1] == 2
    assert inspection["visitation"]["state_action_counts"][1][2][3] == 1


def test_api_offline_rnd_analysis(monkeypatch, tmp_path) -> None:
    run_root = tmp_path / "runs"
    monkeypatch.setenv("RLFLOW_DB_PATH", str(tmp_path / "rlflow.db"))
    monkeypatch.setenv("RLFLOW_RUN_ROOT", str(run_root))
    dataset_path = run_root / "sample" / "replay.npz"
    dataset_path.parent.mkdir(parents=True)
    import numpy as np

    observations = np.stack(
        [
            _symbolic_navix_observation(1, 1),
            _symbolic_navix_observation(1, 2),
            _symbolic_navix_observation(2, 2),
            _symbolic_navix_observation(1, 1),
        ]
    )
    np.savez_compressed(
        dataset_path,
        observations=observations,
        actions=np.asarray([0, 3, 1, 0], dtype=np.int32),
        rewards=np.zeros((4,), dtype=np.float32),
        next_observations=observations,
        terminals=np.asarray([False, False, False, True], dtype=np.bool_),
    )
    client = TestClient(create_app())

    response = client.post(
        "/datasets/offline-rnd",
        json={
            "path": "sample/replay",
            "granularity": "state_action",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "hidden_units": [8],
            "output_dim": 4,
            "seed": 0,
        },
    )

    assert response.status_code == 200
    analysis = response.json()
    assert analysis["unique_items"] == 3
    assert analysis["algorithm"] == "rnd"
    assert len(analysis["loss_history"]) == 1
    assert analysis["learned_state_action_bonus"][1][1][0] is not None
    assert abs(analysis["count_state_action_bonus"][1][1][0] - 1 / (2**0.5)) < 1e-7

    for algorithm in ("cfn", "classifier", "simhash"):
        response = client.post(
            "/datasets/offline-rnd",
            json={
                "path": "sample/replay",
                "algorithm": algorithm,
                "granularity": "state",
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "hidden_units": [8],
                "output_dim": 4,
                "seed": 0,
                "simhash_mode": "learned",
                "simhash_bits": 8,
                "simhash_table_size": 32,
            },
        )
        assert response.status_code == 200
        analysis = response.json()
        assert analysis["algorithm"] == algorithm
        assert len(analysis["loss_history"]) == 1
        assert analysis["learned_state_bonus"][1][1] is not None


def _symbolic_navix_observation(row: int, col: int, size: int = 5):
    import numpy as np

    raw = np.zeros((size, size, 3), dtype=np.float32)
    raw[..., 0] = 1
    raw[0, :, :] = np.asarray([2, 5, 0], dtype=np.float32)
    raw[-1, :, :] = np.asarray([2, 5, 0], dtype=np.float32)
    raw[:, 0, :] = np.asarray([2, 5, 0], dtype=np.float32)
    raw[:, -1, :] = np.asarray([2, 5, 0], dtype=np.float32)
    raw[row, col, :] = np.asarray([10, 0, 0], dtype=np.float32)
    return (raw / 255.0).reshape(-1).astype(np.float32)
