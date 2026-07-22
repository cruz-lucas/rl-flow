"""End-to-end proof that a third-party component is *executable*, not just visible.

Simulates an out-of-tree plugin (``tests/fixtures/external_agent``): its
ComponentSpec is registered the same way entry-point discovery would, the
workflow validates and compiles against the standard registry machinery, and the
builtin runner delegates execution through the ``compile_target`` runtime hook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from rlflow.analysis.run_report import summarize_run
from rlflow.graph.compiler import WorkflowCompiler
from rlflow.graph.validation import WorkflowValidator
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import WorkflowSpec

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "external_agent"


def _external_registry():
    # Make the "package" importable and register its components, exactly as
    # `rlflow.components` entry-point discovery would after `pip install`.
    if str(FIXTURE_DIR) not in sys.path:
        sys.path.insert(0, str(FIXTURE_DIR))
    import example_agent

    registry = create_default_registry(discover=False)
    registry.register_many(example_agent.components())
    return registry


def _workflow() -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "name": "external_agent_riverswim",
            "nodes": [
                {
                    "id": "env",
                    "component": "builtin.env.riverswim",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "num_states": 6,
                        "start_state": 0,
                        "random_start": False,
                        "p_left": 0.1,
                        "p_stay": 0.6,
                        "p_right": 0.3,
                        "easy_reward": 0.005,
                        "hard_reward": 1.0,
                        "common_reward": 0.0,
                    },
                },
                {
                    "id": "agent",
                    "component": "example.agent.random_tabular",
                    "position": {"x": 0, "y": 120},
                    "config": {"episodes": 4},
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 240, "y": 60},
                    "config": {
                        "seed": 0,
                        "train_episodes": 4,
                        "train_steps": None,
                        "max_episode_steps": 20,
                        "eval_episodes": 0,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {
                    "from_node": "env",
                    "from_port": "environment",
                    "to_node": "runner",
                    "to_port": "environment",
                },
                {
                    "from_node": "agent",
                    "from_port": "agent",
                    "to_node": "runner",
                    "to_port": "agent",
                },
            ],
        }
    )


def test_external_agent_validates_compiles_and_executes(tmp_path: Path) -> None:
    registry = _external_registry()
    workflow = _workflow()

    result = WorkflowValidator(registry).validate(workflow)
    assert result.valid, [error.message for error in result.errors]

    WorkflowCompiler(registry).compile(workflow, out_dir=tmp_path)
    assert (tmp_path / "command.sh").exists()

    from rlflow_builtin.runners.tabular_jax import run_compiled

    exit_code = run_compiled(
        workflow_path=tmp_path / "workflow.yaml",
        resolved_config_path=tmp_path / "resolved_config.yaml",
        run_dir=tmp_path,
        registry=registry,
    )
    assert exit_code == 0

    # The external delegate owned the run and wrote the standard outputs.
    metrics = json.loads((tmp_path / "summaries" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["agent"] == "example.agent.random_tabular"
    assert metrics["train_episodes"] == 4
    assert metrics["mean_train_return"] is not None

    history_lines = (tmp_path / "logs" / "train_history.jsonl").read_text().splitlines()
    assert len(history_lines) == 4

    # And the run integrates with the analysis tooling (rlflow report).
    summary = summarize_run(tmp_path)
    assert summary["train_episodes"] == 4
    assert summary["metrics"]["agent"] == "example.agent.random_tabular"


def test_unknown_agent_without_runtime_hook_fails_helpfully(tmp_path: Path) -> None:
    registry = _external_registry()
    workflow = _workflow()
    WorkflowCompiler(registry).compile(workflow, out_dir=tmp_path)

    # Strip the runtime hook to simulate a component that forgot to declare it.
    spec = registry.get("example.agent.random_tabular")
    stripped = spec.model_copy(update={"compile_target": {}})
    bare_registry = create_default_registry(discover=False)
    bare_registry.register(stripped)

    from rlflow_builtin.runners.tabular_jax import run_compiled

    try:
        run_compiled(
            workflow_path=tmp_path / "workflow.yaml",
            resolved_config_path=tmp_path / "resolved_config.yaml",
            run_dir=tmp_path,
            registry=bare_registry,
        )
    except ValueError as exc:
        assert "runtime" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a helpful ValueError for the missing runtime hook")


def test_workflow_yaml_roundtrip_preserves_external_component(tmp_path: Path) -> None:
    # The compiled workflow.yaml still references the external component id.
    registry = _external_registry()
    WorkflowCompiler(registry).compile(_workflow(), out_dir=tmp_path)
    compiled = yaml.safe_load((tmp_path / "workflow.yaml").read_text(encoding="utf-8"))
    assert any(node["component"] == "example.agent.random_tabular" for node in compiled["nodes"])
