"""End-to-end proof that the Atari DQN + R-Max + CFN delegate runs.

Uses the synthetic image-env backend (no envpool, CPU JAX) so it runs anywhere.
Validates + compiles the workflow through the standard machinery, then lets the
builtin runner dispatch to the runtime delegate, and checks the standard run
outputs are written with finite numbers.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

from rlflow.analysis.run_report import summarize_run
from rlflow.graph.compiler import WorkflowCompiler
from rlflow.graph.validation import WorkflowValidator
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import WorkflowSpec

ATARI_AGENT = "builtin.agent.dqn_rmax_atari"


def _workflow() -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "name": "atari_synthetic_dqn_rmax_cfn",
            "nodes": [
                {
                    "id": "environment-2",
                    "component": "builtin.env.atari",
                    "config": {
                        "backend": "synthetic",
                        "num_envs": 2,
                        "frame_stack": 4,
                        "img_height": 42,
                        "img_width": 42,
                        "max_episode_steps": 10,
                        "reward_clip": True,
                    },
                },
                {
                    "id": "replay_buffer-3",
                    "component": "builtin.replay.uniform",
                    "config": {
                        "capacity": 500,
                        "batch_size": 8,
                        "min_size": 16,
                        "updates_per_step": 1,
                        "intrinsic_updates_per_step": 1,
                        "q_network_updates_per_step": 1,
                    },
                },
                {
                    "id": "runner-4",
                    "component": "builtin.runner.tabular_jax",
                    "config": {
                        "seed": 0,
                        "train_episodes": 1,
                        "train_steps": 160,
                        "max_episode_steps": 10,
                        "eval_episodes": 0,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
                {
                    "id": "intrinsic_reward-5",
                    "component": "builtin.intrinsic.cfn",
                    "config": {"cfn_output_dim": 16, "cfn_action_conditioning": "output"},
                },
                {
                    "id": "agent-1",
                    "component": ATARI_AGENT,
                    "config": {
                        "hidden_units": [64],
                        "update_frequency": 1,
                        "target_update_frequency": 50,
                        "epsilon_start": 1.0,
                        "epsilon_end": 0.1,
                        "epsilon_decay_steps": 100,
                        "seed": 0,
                    },
                },
            ],
            "edges": [
                {
                    "from_node": "environment-2",
                    "from_port": "environment",
                    "to_node": "runner-4",
                    "to_port": "environment",
                },
                {
                    "from_node": "replay_buffer-3",
                    "from_port": "replay_buffer",
                    "to_node": "runner-4",
                    "to_port": "replay_buffer",
                },
                {
                    "from_node": "agent-1",
                    "from_port": "agent",
                    "to_node": "runner-4",
                    "to_port": "agent",
                },
                {
                    "from_node": "intrinsic_reward-5",
                    "from_port": "intrinsic_reward",
                    "to_node": "agent-1",
                    "to_port": "knownness_signal",
                },
            ],
        }
    )


def test_atari_delegate_validates_compiles_and_executes(tmp_path: Path) -> None:
    registry = create_default_registry()
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

    metrics = json.loads((tmp_path / "summaries" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["agent"] == ATARI_AGENT
    assert metrics["agent_algorithm"] == "dqn_rmax"
    assert metrics["knownness_signal"] == "builtin.intrinsic.cfn"

    history_lines = (tmp_path / "logs" / "train_history.jsonl").read_text().splitlines()
    assert history_lines, "expected at least one completed episode in the history"
    for line in history_lines:
        row = json.loads(line)
        assert set(row) >= {"episode", "env_step", "return", "length", "loss"}
        assert row["loss"] == row["loss"]  # not NaN
        assert row["return"] == row["return"]

    summary = summarize_run(tmp_path)
    assert summary["metrics"]["agent"] == ATARI_AGENT
