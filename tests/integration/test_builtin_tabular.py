import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from rlflow.graph.compiler import WorkflowCompiler
from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import WorkflowEdge, WorkflowNode, WorkflowSpec
from rlflow_builtin.dqn.training import (
    DqnAgentConfig,
    DqnIntrinsicConfig,
    DqnReplayConfig,
    DqnTrainState,
    _clone_params,
    _count_keys,
    _count_direct_bonus,
    _count_raw_bonus,
    _init_mlp,
    _initial_intrinsic_state,
    _initial_replay_state,
    _make_dqn_environment,
    _observe_intrinsic_transition,
    _optimizer,
    _push_replay,
    _q_loss,
    _replay_updates,
    _rmax_action,
    _select_action,
    _simhash_raw_bonus,
    dqn_replay_config,
    run_dqn_training,
)
from rlflow_builtin.tabular.types import RunnerConfig


def test_builtin_tabular_q_learning_riverswim_runs(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path("configs/workflows/tabular_q_learning_riverswim.yaml").read_text(encoding="utf-8"))
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    command = (tmp_path / "command.sh").read_text(encoding="utf-8")
    assert "rlflow_builtin.runners.tabular_jax" in command

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (6, 2)
    assert metrics["agent"] == "builtin.agent.q_learning_tabular"
    assert metrics["policy"] == "builtin.policy.epsilon_greedy"
    assert metrics["environment"] == "builtin.env.riverswim"
    assert metrics["mean_eval_return"] is not None
    assert (tmp_path / "checkpoints" / "q_learning" / "final_checkpoint.npz").exists()


def test_builtin_tabular_q_learning_accepts_optional_replay_buffer(tmp_path: Path) -> None:
    workflow_data = yaml.safe_load(Path("configs/workflows/tabular_q_learning_riverswim.yaml").read_text(encoding="utf-8"))
    workflow_data["name"] = "tabular_q_learning_riverswim_replay"
    for node in workflow_data["nodes"]:
        if node["id"] == "runner":
            node["config"]["train_episodes"] = 30
            node["config"]["eval_episodes"] = 0
            node["config"]["save_final_checkpoint"] = False
    workflow_data["nodes"].append(
        {
            "id": "replay",
            "component": "builtin.replay.tabular_uniform",
            "position": {"x": 360, "y": 260},
            "config": {
                "capacity": 64,
                "batch_size": 4,
                "min_size": 4,
                "updates_per_step": 1,
            },
        }
    )
    workflow_data["edges"].append(
        {
            "from_node": "replay",
            "from_port": "replay_buffer",
            "to_node": "runner",
            "to_port": "replay_buffer",
        }
    )
    workflow = WorkflowSpec.model_validate(workflow_data)

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["replay_buffer"] == "builtin.replay.tabular_uniform"
    assert np.load(tmp_path / "q_table.npy").shape == (6, 2)


def test_builtin_tabular_sarsa_gridworld_runs(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path("configs/workflows/tabular_sarsa_gridworld.yaml").read_text(encoding="utf-8"))
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (16, 4)
    assert metrics["agent"] == "builtin.agent.sarsa_tabular"
    assert metrics["policy"] == "builtin.policy.ucb"
    assert metrics["mean_eval_return"] is not None


def test_builtin_tabular_q_learning_sixarms_runs(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path("configs/workflows/tabular_q_learning_sixarms.yaml").read_text(encoding="utf-8"))
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (7, 6)
    assert metrics["environment"] == "builtin.env.sixarms"
    assert metrics["mean_eval_return"] is not None


def test_builtin_tabular_q_learning_navix_empty_room_runs(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        {
            "name": "tabular_q_learning_navix_empty_room",
            "nodes": [
                {
                    "id": "env",
                    "component": "navix.env.grid",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "env_name": "empty_room",
                        "size": 5,
                        "layout": "fixed",
                        "observation_mode": "tabular",
                        "action_set": "cardinal",
                        "max_steps": 20,
                    },
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.q_learning_tabular",
                    "position": {"x": 0, "y": 100},
                    "config": {"learning_rate": 0.1, "discount": 0.99, "initial_q": 0.0},
                },
                {
                    "id": "policy",
                    "component": "builtin.policy.epsilon_greedy",
                    "position": {"x": 0, "y": 200},
                    "config": {"epsilon": 0.1, "eval_epsilon": 0.0},
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 2,
                        "max_episode_steps": 5,
                        "eval_episodes": 1,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
                {"from_node": "policy", "from_port": "policy", "to_node": "runner", "to_port": "policy"},
            ],
        }
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (9, 4)
    assert metrics["environment"] == "navix.env.grid"
    assert metrics["mean_eval_return"] is not None


def test_builtin_tabular_rmax_navix_empty_room_runs_without_policy(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        {
            "name": "tabular_rmax_navix_empty_room",
            "nodes": [
                {
                    "id": "env",
                    "component": "navix.env.grid",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "env_name": "empty_room",
                        "size": 5,
                        "layout": "fixed",
                        "observation_mode": "tabular",
                        "action_set": "cardinal",
                        "max_steps": 20,
                    },
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.rmax_tabular",
                    "position": {"x": 0, "y": 100},
                    "config": {
                        "discount": 0.99,
                        "known_count_threshold": 1,
                        "rmax_v_max": 1.0,
                        "planning_iterations": 5,
                    },
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 2,
                        "max_episode_steps": 5,
                        "eval_episodes": 0,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
            ],
        }
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    action_counts = np.load(tmp_path / "action_counts.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (9, 4)
    assert action_counts.shape == (9, 4)
    assert np.sum(action_counts) > 0
    assert metrics["agent"] == "builtin.agent.rmax_tabular"
    assert metrics["policy"] is None
    assert metrics["environment"] == "navix.env.grid"


def test_builtin_tabular_rmax_navix_truncation_bootstraps(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        {
            "name": "tabular_rmax_navix_truncation",
            "nodes": [
                {
                    "id": "env",
                    "component": "navix.env.grid",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "env_name": "empty_room",
                        "size": 5,
                        "layout": "fixed",
                        "observation_mode": "tabular",
                        "action_set": "cardinal",
                        "max_steps": 1,
                    },
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.rmax_tabular",
                    "position": {"x": 0, "y": 100},
                    "config": {
                        "discount": 0.99,
                        "known_count_threshold": 1,
                        "rmax_v_max": 1.0,
                        "planning_iterations": 1,
                    },
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 1,
                        "max_episode_steps": 1,
                        "eval_episodes": 0,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
            ],
        }
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    action_counts = np.load(tmp_path / "action_counts.npy")
    visited_q_values = q_table[action_counts > 0]
    assert visited_q_values.shape == (1,)
    np.testing.assert_allclose(visited_q_values[0], 0.99, rtol=1e-6)


def test_builtin_tabular_q_learning_navix_four_rooms_runs(tmp_path: Path) -> None:
    workflow = WorkflowSpec.model_validate(
        {
            "name": "tabular_q_learning_navix_four_rooms",
            "nodes": [
                {
                    "id": "env",
                    "component": "navix.env.grid",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "env_name": "four_rooms",
                        "size": 19,
                        "layout": "fixed",
                        "observation_mode": "tabular",
                        "action_set": "cardinal",
                        "max_steps": 20,
                    },
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.q_learning_tabular",
                    "position": {"x": 0, "y": 100},
                    "config": {"learning_rate": 0.1, "discount": 0.99, "initial_q": 0.0},
                },
                {
                    "id": "policy",
                    "component": "builtin.policy.epsilon_greedy",
                    "position": {"x": 0, "y": 200},
                    "config": {"epsilon": 0.1, "eval_epsilon": 0.0},
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 1,
                        "max_episode_steps": 4,
                        "eval_episodes": 1,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
                {"from_node": "policy", "from_port": "policy", "to_node": "runner", "to_port": "policy"},
            ],
        }
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    q_table = np.load(tmp_path / "q_table.npy")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert q_table.shape == (289, 4)
    assert metrics["environment"] == "navix.env.grid"
    assert metrics["mean_eval_return"] is not None


def test_builtin_tabular_replay_dataset_can_be_saved_and_loaded_for_offline_rl(tmp_path: Path) -> None:
    collector = _riverswim_dataset_workflow(
        name="collect_riverswim_dataset",
        replay_config={
            "capacity": 64,
            "batch_size": 4,
            "min_size": 1,
            "updates_per_step": 0,
            "save_dataset_path": "datasets/replay.npz",
            "load_dataset_path": "",
            "offline_only": False,
            "offline_updates": 0,
        },
    )
    collect_dir = tmp_path / "collect"
    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(collector, out_dir=collect_dir)

    subprocess.run(["bash", experiment.command], check=True)

    dataset_path = collect_dir / "datasets" / "replay.npz"
    assert dataset_path.exists()
    dataset = np.load(dataset_path)
    assert set(dataset.files) == {"observations", "actions", "rewards", "next_observations", "terminals"}
    assert dataset["observations"].shape == (10,)

    offline = _riverswim_dataset_workflow(
        name="offline_riverswim_dataset",
        replay_config={
            "capacity": 64,
            "batch_size": 4,
            "min_size": 1,
            "updates_per_step": 0,
            "save_dataset_path": "",
            "load_dataset_path": str(dataset_path),
            "offline_only": True,
            "offline_updates": 8,
        },
    )
    offline_dir = tmp_path / "offline"
    offline_experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(offline, out_dir=offline_dir)

    subprocess.run(["bash", offline_experiment.command], check=True)

    metrics = json.loads((offline_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["offline_only"] is True
    assert metrics["loaded_replay_dataset_path"] == str(dataset_path)
    assert np.load(offline_dir / "q_table.npy").shape == (6, 2)


def test_builtin_jax_runner_uses_builtin_dqn_for_non_tabular_navix_observations(tmp_path: Path) -> None:
    workflow = _dqn_navix_workflow(
        observation_mode="symbolic",
        agent_config={
            "learning_rate": 0.001,
            "discount": 0.99,
            "hidden_units": [16],
            "update_frequency": 1,
            "target_update_frequency": 10,
            "epsilon_start": 0.2,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 10,
            "eval_epsilon": 0.0,
            "loss_type": "mse",
        },
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["agent"] == "builtin.agent.dqn_jax"
    assert metrics["runner"] == "builtin.runner.tabular_jax"
    assert metrics["source_observation_shape"] == [5, 5, 3]
    assert metrics["input_dim"] == 75
    assert metrics["num_actions"] == 4
    assert metrics["mean_eval_return"] is not None


def test_builtin_jax_runner_uses_builtin_dqn_for_tabular_navix_observations(tmp_path: Path) -> None:
    workflow = _dqn_navix_workflow(
        observation_mode="tabular",
        agent_config={
            "learning_rate": 0.001,
            "discount": 0.99,
            "hidden_units": [16],
            "update_frequency": 1,
            "target_update_frequency": 10,
            "epsilon_start": 0.2,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 10,
            "eval_epsilon": 0.0,
            "loss_type": "huber",
        },
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["agent"] == "builtin.agent.dqn_jax"
    assert metrics["runner"] == "builtin.runner.tabular_jax"
    assert metrics["source_observation_shape"] == []
    assert metrics["input_dim"] == 9
    assert metrics["num_actions"] == 4
    assert metrics["mean_eval_return"] is not None


def test_builtin_dqn_replay_dataset_save_path_appends_npz(tmp_path: Path) -> None:
    workflow = _dqn_navix_workflow(
        observation_mode="symbolic",
        agent_config={
            "learning_rate": 0.001,
            "discount": 0.99,
            "hidden_units": [16],
            "update_frequency": 1,
            "target_update_frequency": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 1.0,
            "epsilon_decay_steps": 10,
            "eval_epsilon": 0.0,
            "loss_type": "mse",
        },
    )
    for node in workflow.nodes:
        if node.id == "replay":
            node.config["save_dataset_path"] = "datasets/replay"

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    dataset_path = tmp_path / "datasets" / "replay.npz"
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert dataset_path.exists()
    assert metrics["saved_replay_dataset_path"] == str(dataset_path)
    dataset = np.load(dataset_path)
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    assert observations.shape[1:] == (5, 5, 3)
    assert next_observations.shape == observations.shape
    assert observations.dtype == np.uint8
    assert int(observations.max()) == 10
    assert observations[0, 0, 0].tolist() == [2, 5, 0]


def test_builtin_dqn_navix_symbolic_encoder_respects_normalize_observations() -> None:
    settings = {
        "env_name": "empty_room",
        "size": 5,
        "layout": "fixed",
        "observation_mode": "symbolic",
        "action_set": "cardinal",
        "max_steps": 20,
    }
    raw_env = _make_dqn_environment(
        "navix.env.grid",
        settings,
        normalize_observations=False,
    )
    normalized_env = _make_dqn_environment(
        "navix.env.grid",
        settings,
        normalize_observations=True,
    )
    timestep = raw_env.reset(jax.random.PRNGKey(0))

    raw_encoded = np.asarray(raw_env.encode(timestep.observation))
    normalized_encoded = np.asarray(normalized_env.encode(timestep.observation))

    assert raw_encoded.shape == (75,)
    assert float(raw_encoded.max()) == 10.0
    assert np.isclose(float(normalized_encoded.max()), 10.0 / 255.0)


def test_builtin_dqn_navix_exposes_oracle_tabular_state_ids() -> None:
    env = _make_dqn_environment(
        "navix.env.grid",
        {
            "env_name": "empty_room",
            "size": 5,
            "layout": "fixed",
            "observation_mode": "symbolic",
            "action_set": "cardinal",
            "max_steps": 20,
            "symbolic_distractor": "corner_wall_color",
        },
    )
    timestep = env.reset(jax.random.PRNGKey(0))

    assert env.oracle_state_id is not None
    assert env.oracle_state_space_size == 9
    assert int(np.asarray(env.oracle_state_id(timestep))) == 0


def test_builtin_dqn_navix_four_rooms_exposes_oracle_tabular_state_ids() -> None:
    env = _make_dqn_environment(
        "navix.env.grid",
        {
            "env_name": "four_rooms",
            "size": 19,
            "layout": "fixed",
            "observation_mode": "symbolic",
            "action_set": "cardinal",
            "max_steps": 20,
            "symbolic_distractor": "corner_wall_color",
        },
    )
    timestep = env.reset(jax.random.PRNGKey(0))

    assert env.oracle_state_id is not None
    assert env.oracle_state_space_size == 289
    assert int(np.asarray(env.oracle_state_id(timestep))) == 0


def test_builtin_jax_runner_uses_dqn_rmax_with_count_bonus(tmp_path: Path) -> None:
    workflow = _dqn_navix_workflow(
        observation_mode="tabular",
        agent_config={
            "learning_rate": 0.001,
            "discount": 0.99,
            "hidden_units": [16],
            "update_frequency": 1,
            "target_update_frequency": 4,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "eval_epsilon": 0.0,
            "loss_type": "mse",
            "rmax_bonus_threshold": 0.5,
            "rmax_decision_v_max": 100.0,
            "rmax_update_v_max": 100.0,
        },
    )
    for node in workflow.nodes:
        if node.id == "agent":
            node.component = "builtin.agent.dqn_rmax_jax"
        if node.id == "replay":
            node.config["batch_size"] = 2
            node.config["min_size"] = 1
        if node.id == "runner":
            node.config["max_episode_steps"] = 3
    workflow.nodes.append(
        WorkflowNode.model_validate(
            {
                "id": "intrinsic",
                "component": "builtin.intrinsic.count",
                "position": {"x": 0, "y": 300},
                "config": {
                    "intrinsic_reward_scale": 1.0,
                    "intrinsic_stats_decay": 1.0,
                    "intrinsic_reward_epsilon": 1e-4,
                    "intrinsic_reward_clip": 10.0,
                    "intrinsic_reward_center": False,
                    "count_action_conditioning": "input",
                    "count_key_mode": "oracle_tabular",
                    "count_table_size": 0,
                    "count_bonus_exponent": 0.5,
                    "count_min_count": 1.0,
                },
            }
        )
    )
    workflow.edges.append(
        WorkflowEdge.model_validate(
            {
                "from_node": "intrinsic",
                "from_port": "intrinsic_reward",
                "to_node": "agent",
                "to_port": "knownness_signal",
            }
        )
    )
    workflow.edges.append(
        WorkflowEdge.model_validate(
            {
                "from_node": "intrinsic",
                "from_port": "intrinsic_reward",
                "to_node": "runner",
                "to_port": "intrinsic_reward",
            }
        )
    )

    experiment = WorkflowCompiler(create_default_registry(discover=False)).compile(workflow, out_dir=tmp_path)

    subprocess.run(["bash", experiment.command], check=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["agent"] == "builtin.agent.dqn_rmax_jax"
    assert metrics["agent_algorithm"] == "dqn_rmax"
    assert metrics["intrinsic_reward"] == "builtin.intrinsic.count"
    assert metrics["knownness_signal"] == "builtin.intrinsic.count"
    assert metrics["intrinsic_reward_shared_with_knownness"] is True
    assert metrics["count_table_entries"] is not None
    assert metrics["count_table_overflow"] is False
    assert metrics["mean_eval_return"] is not None


def test_count_bonus_uses_exact_limited_table() -> None:
    agent = _minimal_dqn_agent()
    intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_table_size=2,
        count_bonus_exponent=0.5,
        count_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=2,
        key=jax.random.PRNGKey(0),
    )
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(0), intrinsic, 2)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(0), intrinsic, 2)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(1), intrinsic, 2)

    assert int(np.asarray(state.count_size)) == 2
    assert bool(np.asarray(state.count_overflow)) is False
    bonuses = _count_raw_bonus(
        state,
        jnp.stack([observation, observation]),
        jnp.asarray([0, 1], dtype=jnp.int32),
        intrinsic,
        2,
    )
    np.testing.assert_allclose(
        np.asarray(bonuses),
        np.asarray([1.0 / np.sqrt(2.0), 1.0]),
        rtol=1e-6,
    )

    state = _observe_intrinsic_transition(
        state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        jnp.asarray(0),
        intrinsic,
        2,
    )
    assert int(np.asarray(state.count_size)) == 2
    assert bool(np.asarray(state.count_overflow)) is True


def test_oracle_tabular_count_uses_direct_state_action_indices() -> None:
    agent = _minimal_dqn_agent(algorithm="dqn_rmax")
    intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_key_mode="oracle_tabular",
        count_table_size=0,
        count_bonus_exponent=0.5,
        count_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=75,
        num_actions=4,
        key=jax.random.PRNGKey(0),
        oracle_state_space_size=9,
    )
    observation = jnp.zeros((75,), dtype=jnp.float32)
    state_id = jnp.asarray(4, dtype=jnp.int32)

    state = _observe_intrinsic_transition(
        state,
        observation,
        jnp.asarray(2),
        intrinsic,
        4,
        state_id=state_id,
    )
    state = _observe_intrinsic_transition(
        state,
        observation,
        jnp.asarray(2),
        intrinsic,
        4,
        state_id=state_id,
    )

    assert state.counts.shape == (36,)
    assert state.count_keys.shape == (36, 1)
    assert int(np.asarray(state.count_size)) == 0
    assert float(np.asarray(state.counts[4 * 4 + 2])) == 2.0

    bonuses = _count_direct_bonus(
        state,
        jnp.asarray([4, 4], dtype=jnp.int32),
        jnp.asarray([2, 1], dtype=jnp.int32),
        intrinsic,
        4,
    )
    np.testing.assert_allclose(
        np.asarray(bonuses),
        np.asarray([1.0 / np.sqrt(2.0), 1.0]),
        rtol=1e-6,
    )


def test_simhash_bonus_uses_static_hash_counts() -> None:
    agent = _minimal_dqn_agent()
    intrinsic = DqnIntrinsicConfig(
        kind="simhash",
        action_conditioning="input",
        simhash_mode="static",
        simhash_bits=8,
        simhash_table_size=4,
        simhash_bonus_exponent=0.5,
        simhash_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=2,
        key=jax.random.PRNGKey(0),
    )
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(0), intrinsic, 2)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(0), intrinsic, 2)

    bonuses = _simhash_raw_bonus(
        state,
        jnp.stack([observation, observation]),
        jnp.asarray([0, 1], dtype=jnp.int32),
        intrinsic,
        2,
    )

    assert int(np.asarray(state.count_size)) == 1
    np.testing.assert_allclose(float(np.asarray(bonuses[0])), 1.0 / np.sqrt(2.0), rtol=1e-6)
    assert float(np.asarray(bonuses[1])) >= 1.0 / np.sqrt(2.0)


def test_simhash_learned_initializes_autoencoder_features() -> None:
    agent = _minimal_dqn_agent()
    intrinsic = DqnIntrinsicConfig(
        kind="simhash",
        hidden_units=(4,),
        action_conditioning="input",
        simhash_mode="learned",
        simhash_bits=6,
        output_dim=3,
        simhash_table_size=4,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=2,
        key=jax.random.PRNGKey(0),
    )
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    state = _observe_intrinsic_transition(state, observation, jnp.asarray(0), intrinsic, 2)

    assert state.target_params[0]["w"].shape == (3, 6)
    assert int(np.asarray(state.count_size)) == 1


def test_count_keys_can_ignore_empty_room_symbolic_distractor() -> None:
    distractor_observation = _symbolic_navix_observation(1, 1, wall_colour=12)
    other_distractor_observation = _symbolic_navix_observation(1, 1, wall_colour=42)
    other_state_observation = _symbolic_navix_observation(1, 2, wall_colour=12)
    observation_batch = np.stack(
        [
            distractor_observation,
            other_distractor_observation,
            other_state_observation,
        ]
    )
    actions = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    exact_intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_ignore_empty_room_distractor=False,
    )
    ignore_intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_ignore_empty_room_distractor=True,
    )

    for observations_np in (observation_batch, observation_batch / 255.0):
        observations = jnp.asarray(observations_np, dtype=jnp.float32)
        exact_keys = _count_keys(observations, actions, exact_intrinsic, 4)
        ignore_keys = _count_keys(observations, actions, ignore_intrinsic, 4)

        assert not np.allclose(np.asarray(exact_keys[0]), np.asarray(exact_keys[1]))
        np.testing.assert_allclose(np.asarray(ignore_keys[0]), np.asarray(ignore_keys[1]))
        assert not np.allclose(np.asarray(ignore_keys[0]), np.asarray(ignore_keys[2]))


def test_rmax_breaks_ties_randomly() -> None:
    agent = _minimal_dqn_agent(algorithm="dqn_rmax")
    intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_table_size=8,
        count_bonus_exponent=0.5,
        count_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=3,
        key=jax.random.PRNGKey(0),
    )
    params = ({"w": jnp.zeros((2, 3), dtype=jnp.float32), "b": jnp.zeros((3,), dtype=jnp.float32)},)
    actions = {
        int(
            np.asarray(
                _rmax_action(
                    agent,
                    params,
                    state,
                    intrinsic,
                    jnp.asarray([1.0, 0.0], dtype=jnp.float32),
                    jax.random.PRNGKey(seed),
                    3,
                )
            )
        )
        for seed in range(20)
    }

    assert len(actions) > 1


def test_dqn_select_action_breaks_q_value_ties_randomly() -> None:
    agent = _minimal_dqn_agent()
    intrinsic = DqnIntrinsicConfig()
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=3,
        key=jax.random.PRNGKey(0),
    )
    params = ({"w": jnp.zeros((2, 3), dtype=jnp.float32), "b": jnp.zeros((3,), dtype=jnp.float32)},)
    actions = {
        int(
            np.asarray(
                _select_action(
                    agent,
                    params,
                    state,
                    intrinsic,
                    jnp.asarray([1.0, 0.0], dtype=jnp.float32),
                    jax.random.PRNGKey(seed),
                    3,
                    jnp.asarray(0, dtype=jnp.int32),
                    training=True,
                )
            )
        )
        for seed in range(20)
    }

    assert len(actions) > 1


def test_double_dqn_target_breaks_q_value_ties_randomly() -> None:
    base_agent = _minimal_dqn_agent()
    agent = DqnAgentConfig(**{**base_agent.__dict__, "double_q": True, "discount": 1.0})
    params = ({"w": jnp.zeros((2, 3), dtype=jnp.float32), "b": jnp.zeros((3,), dtype=jnp.float32)},)
    target_params = (
        {
            "w": jnp.zeros((2, 3), dtype=jnp.float32),
            "b": jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        },
    )
    losses = {
        float(
            np.asarray(
                _q_loss(
                    agent,
                    params,
                    target_params,
                    observations=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
                    actions=jnp.asarray([0], dtype=jnp.int32),
                    rewards=jnp.asarray([0.0], dtype=jnp.float32),
                    next_observations=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
                    terminals=jnp.asarray([0.0], dtype=jnp.float32),
                    known_mask=jnp.asarray([True]),
                    next_unknown_any=jnp.asarray([False]),
                    key=jax.random.PRNGKey(seed),
                )
            )
        )
        for seed in range(20)
    }

    assert len(losses) > 1


def test_rmax_uses_separate_decision_and_update_v_max() -> None:
    agent = _minimal_dqn_agent(algorithm="dqn_rmax")
    agent = DqnAgentConfig(
        **{
            **agent.__dict__,
            "discount": 1.0,
            "rmax_decision_v_max": 5.0,
            "rmax_update_v_max": 7.0,
        }
    )
    intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_table_size=8,
        count_bonus_exponent=1.0,
        count_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=3,
        key=jax.random.PRNGKey(0),
    )
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    for action in (0, 2):
        for _ in range(4):
            state = _observe_intrinsic_transition(
                state,
                observation,
                jnp.asarray(action, dtype=jnp.int32),
                intrinsic,
                3,
            )

    params = (
        {
            "w": jnp.zeros((2, 3), dtype=jnp.float32),
            "b": jnp.asarray([0.0, 2.0, 3.0], dtype=jnp.float32),
        },
    )
    action = _rmax_action(agent, params, state, intrinsic, observation, jax.random.PRNGKey(0), 3)
    assert int(np.asarray(action)) == 1

    loss = _q_loss(
        agent,
        params,
        params,
        observations=observation[None, :],
        actions=jnp.asarray([0], dtype=jnp.int32),
        rewards=jnp.asarray([0.0], dtype=jnp.float32),
        next_observations=observation[None, :],
        terminals=jnp.asarray([0.0], dtype=jnp.float32),
        known_mask=jnp.asarray([True]),
        next_unknown_any=jnp.asarray([True]),
        key=jax.random.PRNGKey(0),
    )
    np.testing.assert_allclose(np.asarray(loss), 49.0)


def test_rmax_select_action_uses_epsilon_greedy_with_rmax_greedy_action() -> None:
    base_agent = _minimal_dqn_agent(algorithm="dqn_rmax")
    agent = DqnAgentConfig(
        **{
            **base_agent.__dict__,
            "epsilon_start": 1.0,
            "epsilon_end": 1.0,
            "epsilon_decay_steps": 1,
            "rmax_decision_v_max": 5.0,
        }
    )
    intrinsic = DqnIntrinsicConfig(
        kind="count",
        action_conditioning="input",
        count_table_size=8,
        count_bonus_exponent=1.0,
        count_min_count=1.0,
    )
    state = _initial_intrinsic_state(
        agent,
        intrinsic,
        input_dim=2,
        num_actions=3,
        key=jax.random.PRNGKey(0),
    )
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    for action in (0, 2):
        for _ in range(4):
            state = _observe_intrinsic_transition(
                state,
                observation,
                jnp.asarray(action, dtype=jnp.int32),
                intrinsic,
                3,
            )

    params = (
        {
            "w": jnp.zeros((2, 3), dtype=jnp.float32),
            "b": jnp.asarray([0.0, 2.0, 3.0], dtype=jnp.float32),
        },
    )
    greedy_agent = DqnAgentConfig(
        **{
            **agent.__dict__,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
        }
    )

    greedy_action = _select_action(
        greedy_agent,
        params,
        state,
        intrinsic,
        observation,
        jax.random.PRNGKey(0),
        3,
        jnp.asarray(0, dtype=jnp.int32),
        training=True,
    )
    exploratory_action = _select_action(
        agent,
        params,
        state,
        intrinsic,
        observation,
        jax.random.PRNGKey(4),
        3,
        jnp.asarray(0, dtype=jnp.int32),
        training=True,
    )

    assert int(np.asarray(greedy_action)) == 1
    assert int(np.asarray(exploratory_action)) == 0


def test_builtin_dqn_training_accepts_train_steps() -> None:
    result = run_dqn_training(
        env_component="builtin.env.riverswim",
        env_settings={
            "num_states": 6,
            "start_state": 1,
            "random_start": False,
            "p_left": 0.1,
            "p_stay": 0.6,
            "p_right": 0.3,
            "easy_reward": 5.0,
            "hard_reward": 10000.0,
            "common_reward": 0.0,
        },
        agent=_minimal_dqn_agent(),
        replay=DqnReplayConfig(
            name="builtin.replay.uniform",
            capacity=8,
            batch_size=2,
            min_size=100,
            updates_per_step=1,
        ),
        runner=RunnerConfig(
            seed=0,
            train_episodes=99,
            train_steps=3,
            max_episode_steps=10,
            eval_episodes=0,
            checkpoint_freq=None,
            checkpoint_dir="checkpoints",
            save_final_checkpoint=False,
        ),
    )

    assert int(result.train_lengths.sum()) == 3


def test_dqn_replay_config_split_update_defaults_inherit_updates_per_step() -> None:
    replay = dqn_replay_config(
        "builtin.replay.uniform",
        {
            "capacity": 8,
            "batch_size": 2,
            "min_size": 1,
            "updates_per_step": 4,
            "intrinsic_updates_per_step": None,
            "q_network_updates_per_step": None,
        },
    )

    assert replay.intrinsic_updates_per_step == 4
    assert replay.q_network_updates_per_step == 4

    replay = dqn_replay_config(
        "builtin.replay.uniform",
        {
            "capacity": 8,
            "batch_size": 2,
            "min_size": 1,
            "updates_per_step": 4,
            "intrinsic_updates_per_step": 3,
            "q_network_updates_per_step": 2,
        },
    )

    assert replay.intrinsic_updates_per_step == 3
    assert replay.q_network_updates_per_step == 2


def test_dqn_replay_updates_split_intrinsic_and_q_counts() -> None:
    agent = _minimal_dqn_agent()
    intrinsic = DqnIntrinsicConfig(
        kind="rnd",
        action_conditioning="none",
        hidden_units=(),
        output_dim=1,
        learning_rate=agent.learning_rate,
    )
    q_optimizer = _optimizer(agent, agent.learning_rate, agent.optimizer)
    knownness = DqnIntrinsicConfig()
    intrinsic_optimizer = _optimizer(agent, knownness.learning_rate, knownness.optimizer)
    reward_intrinsic_optimizer = _optimizer(
        agent,
        intrinsic.learning_rate,
        intrinsic.optimizer,
    )
    q_params = _init_mlp(jax.random.PRNGKey(1), 2, agent.hidden_units, 2)
    replay_state = _initial_replay_state(
        capacity=4,
        input_dim=2,
        intrinsic_target_dim=1,
        source_observation_shape=(),
        source_observation_dtype="int32",
    )
    transitions = (
        (
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=jnp.float32),
        ),
        (
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    for index, (observation, next_observation, action, reward) in enumerate(transitions):
        replay_state = _push_replay(
            replay_state,
            observation,
            jnp.asarray(index, dtype=jnp.int32),
            action,
            reward,
            next_observation,
            jnp.asarray(index + 1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index + 1, dtype=jnp.int32),
        )
    state = DqnTrainState(
        params=q_params,
        target_params=_clone_params(q_params),
        opt_state=q_optimizer.init(q_params),
        intrinsic_state=_initial_intrinsic_state(
            agent,
            knownness,
            input_dim=2,
            num_actions=2,
            key=jax.random.PRNGKey(2),
        ),
        reward_intrinsic_state=_initial_intrinsic_state(
            agent,
            intrinsic,
            input_dim=2,
            num_actions=2,
            key=jax.random.PRNGKey(4),
        ),
        replay_state=replay_state,
        key=jax.random.PRNGKey(3),
        global_step=jnp.asarray(0, dtype=jnp.int32),
        gradient_step=jnp.asarray(0, dtype=jnp.int32),
        intrinsic_gradient_step=jnp.asarray(0, dtype=jnp.int32),
        reward_intrinsic_gradient_step=jnp.asarray(0, dtype=jnp.int32),
    )
    replay = DqnReplayConfig(
        name="builtin.replay.uniform",
        capacity=4,
        batch_size=2,
        min_size=1,
        updates_per_step=5,
        intrinsic_updates_per_step=3,
        q_network_updates_per_step=2,
    )

    state, loss = _replay_updates(
        state,
        agent,
        replay,
        knownness,
        intrinsic,
        False,
        q_optimizer,
        intrinsic_optimizer,
        reward_intrinsic_optimizer,
        num_actions=2,
    )

    assert int(np.asarray(state.intrinsic_gradient_step)) == 0
    assert int(np.asarray(state.reward_intrinsic_gradient_step)) == 3
    assert int(np.asarray(state.gradient_step)) == 2
    assert np.isfinite(np.asarray(loss))


def _riverswim_dataset_workflow(name: str, replay_config: dict) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "name": name,
            "nodes": [
                {
                    "id": "env",
                    "component": "builtin.env.riverswim",
                    "position": {"x": 0, "y": 0},
                    "config": {"random_start": True},
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.q_learning_tabular",
                    "position": {"x": 0, "y": 100},
                    "config": {"learning_rate": 0.1, "discount": 0.99, "initial_q": 0.0},
                },
                {
                    "id": "policy",
                    "component": "builtin.policy.epsilon_greedy",
                    "position": {"x": 0, "y": 200},
                    "config": {"epsilon": 0.2, "eval_epsilon": 0.0},
                },
                {
                    "id": "replay",
                    "component": "builtin.replay.tabular_uniform",
                    "position": {"x": 280, "y": 200},
                    "config": replay_config,
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 2,
                        "max_episode_steps": 5,
                        "eval_episodes": 0,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
                {"from_node": "policy", "from_port": "policy", "to_node": "runner", "to_port": "policy"},
                {"from_node": "replay", "from_port": "replay_buffer", "to_node": "runner", "to_port": "replay_buffer"},
            ],
        }
    )


def _minimal_dqn_agent(algorithm: str = "dqn") -> DqnAgentConfig:
    return DqnAgentConfig(
        algorithm=algorithm,
        learning_rate=0.001,
        discount=0.99,
        hidden_units=(),
        activation="relu",
        update_frequency=1,
        target_update_frequency=4,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        eval_epsilon=0.0,
        loss_type="mse",
        huber_delta=1.0,
        double_q=False,
        max_grad_norm=1.0,
        optimizer="adam",
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
        optimizer_epsilon=1e-8,
        optimizer_weight_decay=0.0,
        optimizer_momentum=0.0,
        optimizer_decay=0.95,
        optimizer_centered=False,
        normalize_observations=False,
        obs_normalization_epsilon=1e-8,
        obs_normalization_clip=5.0,
        rmax_bonus_threshold=0.5,
        rmax_decision_v_max=100.0,
        rmax_update_v_max=100.0,
        seed=0,
    )


def _symbolic_navix_observation(row: int, col: int, *, size: int = 5, wall_colour: int) -> np.ndarray:
    raw = np.zeros((size, size, 3), dtype=np.float32)
    raw[..., 0] = 1
    raw[0, :, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[-1, :, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[:, 0, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[:, -1, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[row, col, :] = np.asarray([10, 0, 0], dtype=np.float32)
    return raw.reshape(-1).astype(np.float32)


def _dqn_navix_workflow(
    *,
    observation_mode: str,
    agent_config: dict,
) -> WorkflowSpec:
    return WorkflowSpec.model_validate(
        {
            "name": f"dqn_{observation_mode}_navix",
            "nodes": [
                {
                    "id": "env",
                    "component": "navix.env.grid",
                    "position": {"x": 0, "y": 0},
                    "config": {
                        "env_name": "empty_room",
                        "size": 5,
                        "layout": "fixed",
                        "observation_mode": observation_mode,
                        "action_set": "cardinal",
                        "max_steps": 20,
                    },
                },
                {
                    "id": "agent",
                    "component": "builtin.agent.dqn_jax",
                    "position": {"x": 0, "y": 100},
                    "config": agent_config,
                },
                {
                    "id": "replay",
                    "component": "builtin.replay.uniform",
                    "position": {"x": 0, "y": 200},
                    "config": {
                        "capacity": 128,
                        "batch_size": 2,
                        "min_size": 100,
                        "updates_per_step": 1,
                    },
                },
                {
                    "id": "runner",
                    "component": "builtin.runner.tabular_jax",
                    "position": {"x": 300, "y": 100},
                    "config": {
                        "seed": 0,
                        "train_episodes": 1,
                        "max_episode_steps": 2,
                        "eval_episodes": 1,
                        "checkpoint_freq": None,
                        "checkpoint_dir": "checkpoints",
                        "save_final_checkpoint": False,
                    },
                },
            ],
            "edges": [
                {"from_node": "env", "from_port": "environment", "to_node": "runner", "to_port": "environment"},
                {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
                {"from_node": "replay", "from_port": "replay_buffer", "to_node": "runner", "to_port": "replay_buffer"},
            ],
        }
    )
