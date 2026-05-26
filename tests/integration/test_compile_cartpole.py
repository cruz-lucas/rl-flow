import subprocess
from pathlib import Path

import yaml

from rlflow.graph.compiler import WorkflowCompiler
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import WorkflowSpec


def test_compile_builtin_dqn_navix_fixture(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path("configs/workflows/navix_dqn_empty_room.yaml").read_text(encoding="utf-8"))
    )
    for node in workflow.nodes:
        if node.id == "runner":
            node.config["train_episodes"] = 1
            node.config["max_episode_steps"] = 2
            node.config["eval_episodes"] = 1
            node.config["save_final_checkpoint"] = False

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    assert (tmp_path / "workflow.yaml").exists()
    assert (tmp_path / "resolved_config.yaml").exists()
    assert (tmp_path / "generated.gin").exists()
    assert (tmp_path / "command.sh").exists()
    assert "rlflow_builtin.runners.tabular_jax" in (tmp_path / "command.sh").read_text(encoding="utf-8")
    assert experiment.command.endswith("command.sh")

    subprocess.run(["bash", str(tmp_path / "command.sh")], check=True)
    assert (tmp_path / "metrics.json").exists()
