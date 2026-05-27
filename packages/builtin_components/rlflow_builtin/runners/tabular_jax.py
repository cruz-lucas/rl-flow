from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml

from rlflow.registry.builtin import create_default_registry
from rlflow.schemas.workflow import WorkflowSpec
from rlflow_builtin.dqn.training import (
    DQN_AGENT_COMPONENT,
    DQN_RMAX_AGENT_COMPONENT,
    DqnReplayConfig,
    dqn_agent_config,
    dqn_intrinsic_config,
    dqn_replay_config,
    run_dqn_training,
)
from rlflow_builtin.tabular.runtime import (
    agent_config,
    buffer_config,
    environment_config,
    no_buffer_config,
    policy_config,
    resolve_buffer_paths,
    run_tabular_training,
    runner_config,
    save_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="rl-flow builtin JAX runner")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--gin_file", required=False)
    parser.add_argument("--resolved_config", required=True)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    workflow = WorkflowSpec.model_validate(
        yaml.safe_load(Path(args.workflow).read_text(encoding="utf-8"))
    )
    resolved = yaml.safe_load(Path(args.resolved_config).read_text(encoding="utf-8"))
    registry = create_default_registry(discover=False)

    node_ids = _runner_inputs(workflow, registry)
    agent_node = workflow_node(workflow, node_ids["agent"])
    env_node = workflow_node(workflow, node_ids["environment"])
    runner_node = workflow_node(workflow, node_ids["runner"])
    buffer_node = (
        workflow_node(workflow, node_ids["replay_buffer"])
        if "replay_buffer" in node_ids
        else None
    )
    intrinsic_node = (
        workflow_node(workflow, node_ids["intrinsic_reward"])
        if "intrinsic_reward" in node_ids
        else None
    )

    runner = runner_config(resolved[runner_node.id])
    if agent_node.component in {DQN_AGENT_COMPONENT, DQN_RMAX_AGENT_COMPONENT}:
        if buffer_node is None:
            raise ValueError("builtin DQN agents require a builtin.replay.uniform input")
        agent = dqn_agent_config(agent_node.component, resolved[agent_node.id])
        replay = _dqn_replay_from_node(buffer_node, resolved)
        intrinsic = (
            dqn_intrinsic_config(intrinsic_node.component, resolved[intrinsic_node.id], agent)
            if intrinsic_node is not None
            else dqn_intrinsic_config(None, None, agent)
        )
        result = run_dqn_training(
            env_component=env_node.component,
            env_settings=resolved[env_node.id],
            agent=agent,
            runner=runner,
            replay=replay,
            intrinsic=intrinsic,
            run_dir=run_dir,
        )
        _write_dqn_outputs(
            run_dir=run_dir,
            result=result,
            agent_component=agent_node.component,
            env_component=env_node.component,
            runner_component=runner_node.component,
            runner_settings=resolved[runner_node.id],
            replay_component=None if buffer_node is None else buffer_node.component,
            replay=replay,
            intrinsic_component=None if intrinsic_node is None else intrinsic_node.component,
        )
        return 0

    if "policy" not in node_ids:
        raise ValueError("builtin tabular agents require a policy input")

    policy_node = workflow_node(workflow, node_ids["policy"])
    agent = agent_config(agent_node.component, resolved[agent_node.id])
    policy = policy_config(policy_node.component, resolved[policy_node.id])
    environment = environment_config(env_node.component, resolved[env_node.id])
    replay_buffer = (
        buffer_config(buffer_node.component, resolved[buffer_node.id])
        if buffer_node is not None
        else no_buffer_config()
    )
    replay_buffer = resolve_buffer_paths(replay_buffer, run_dir)
    result = run_tabular_training(agent, policy, environment, runner, replay_buffer)
    _write_tabular_outputs(
        run_dir=run_dir,
        result=result,
        agent=agent,
        agent_component=agent_node.component,
        policy_component=policy_node.component,
        env_component=env_node.component,
        replay_component=None if buffer_node is None else buffer_node.component,
        replay_buffer=replay_buffer,
        runner=runner,
        runner_component=runner_node.component,
        runner_settings=resolved[runner_node.id],
    )
    return 0


def _dqn_replay_from_node(
    buffer_node: Any | None,
    resolved: dict[str, dict[str, Any]],
) -> DqnReplayConfig:
    if buffer_node is None:
        raise ValueError("builtin DQN agents require a builtin.replay.uniform input")
    return dqn_replay_config(buffer_node.component, resolved[buffer_node.id])


def _write_tabular_outputs(
    *,
    run_dir: Path,
    result,
    agent,
    agent_component: str,
    policy_component: str,
    env_component: str,
    replay_component: str | None,
    replay_buffer,
    runner,
    runner_component: str,
    runner_settings: dict[str, Any],
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "q_table.npy", result.q_table)
    np.save(run_dir / "action_counts.npy", result.action_counts)
    _write_episode_history(
        logs_dir / "train_history.jsonl",
        result.train_returns,
        result.train_lengths,
        result.train_losses,
    )
    _write_eval_history(
        logs_dir / "eval_history.jsonl",
        result.eval_returns,
        result.eval_lengths,
    )

    dataset_path = None
    if replay_buffer.save_dataset_path and result.dataset is not None:
        dataset_path = replay_buffer.save_dataset_path
        Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            observations=result.dataset.observations,
            actions=result.dataset.actions,
            rewards=result.dataset.rewards,
            next_observations=result.dataset.next_observations,
            terminals=result.dataset.terminals,
        )

    final_checkpoint = None
    if runner.save_final_checkpoint:
        checkpoint_path = (
            run_dir
            / runner.checkpoint_dir
            / agent.algorithm
            / "final_checkpoint.npz"
        )
        save_checkpoint(
            result,
            checkpoint_path,
            {
                "agent": agent_component,
                "policy": policy_component,
                "environment": env_component,
                "runner": runner_component,
            },
        )
        final_checkpoint = str(checkpoint_path)

    summary = {
        "agent": agent_component,
        "environment": env_component,
        "policy": policy_component,
        "replay_buffer": replay_component,
        "runner": runner_component,
        "train_episodes": int(runner_settings["train_episodes"]),
        "train_steps": runner_settings.get("train_steps"),
        **_train_return_metrics(result.train_returns),
        "mean_eval_return": (
            float(np.mean(result.eval_returns)) if len(result.eval_returns) else None
        ),
        "final_checkpoint": final_checkpoint,
        "saved_replay_dataset_path": dataset_path,
        "loaded_replay_dataset_path": replay_buffer.load_dataset_path or None,
        "offline_only": bool(replay_buffer.offline_only),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


def _write_dqn_outputs(
    *,
    run_dir: Path,
    result,
    agent_component: str,
    env_component: str,
    runner_component: str,
    runner_settings: dict[str, Any],
    replay_component: str | None,
    replay: DqnReplayConfig,
    intrinsic_component: str | None,
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _write_episode_history(
        logs_dir / "train_history.jsonl",
        result.train_returns,
        result.train_lengths,
        result.train_losses,
    )
    _write_eval_history(
        logs_dir / "eval_history.jsonl",
        result.eval_returns,
        result.eval_lengths,
    )

    checkpoint_path = None
    if bool(runner_settings.get("save_final_checkpoint", False)):
        checkpoint_path = (
            run_dir
            / str(runner_settings.get("checkpoint_dir", "checkpoints"))
            / "dqn"
            / "final_checkpoint.npz"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            **_flatten_params("q", result.params),
            **{
                key: value
                for name, params in result.aux_params.items()
                for key, value in _flatten_params(name, params).items()
            },
        )

    summary = {
        "agent": agent_component,
        "environment": env_component,
        "policy": None,
        "replay_buffer": replay_component,
        "runner": runner_component,
        "train_episodes": int(runner_settings["train_episodes"]),
        "train_steps": runner_settings.get("train_steps"),
        **_train_return_metrics(result.train_returns),
        "mean_eval_return": (
            float(np.mean(result.eval_returns)) if len(result.eval_returns) else None
        ),
        "final_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "source_observation_shape": list(result.source_observation_shape),
        "source_observation_dtype": result.source_observation_dtype,
        "input_dim": result.input_dim,
        "num_actions": result.num_actions,
        "network": "builtin.mlp_q_network",
        "agent_algorithm": "dqn_rmax" if agent_component == DQN_RMAX_AGENT_COMPONENT else "dqn",
        "intrinsic_reward": intrinsic_component,
        "count_table_entries": result.count_table_entries,
        "count_table_overflow": result.count_table_overflow,
        "saved_replay_dataset_path": (
            str(_resolve_output_path(replay.save_dataset_path, run_dir))
            if replay.save_dataset_path
            else None
        ),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


def _runner_inputs(workflow: WorkflowSpec, registry) -> dict[str, str]:
    runner_nodes = [
        node
        for node in workflow.nodes
        if registry.get(node.component).kind == "runner"
        and node.component == "builtin.runner.tabular_jax"
    ]
    if len(runner_nodes) != 1:
        raise ValueError("Builtin JAX command requires exactly one builtin.runner.tabular_jax node")
    runner = runner_nodes[0]
    inputs = {"runner": runner.id}
    runner_ports = {"agent", "environment", "policy", "replay_buffer", "intrinsic_reward"}
    for edge in workflow.edges:
        if edge.to_node == runner.id and edge.to_port in runner_ports:
            inputs[edge.to_port] = edge.from_node
    missing = {"agent", "environment"} - set(inputs)
    if missing:
        raise ValueError(f"Builtin JAX runner is missing inputs: {sorted(missing)}")
    return inputs


def workflow_node(workflow: WorkflowSpec, node_id: str):
    for node in workflow.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Workflow node not found: {node_id}")


def _write_episode_history(
    path: Path,
    returns: np.ndarray,
    lengths: np.ndarray,
    losses: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, (episode_return, episode_length, loss) in enumerate(
            zip(returns, lengths, losses, strict=True)
        ):
            handle.write(
                json.dumps(
                    {
                        "episode": idx,
                        "return": float(episode_return),
                        "length": int(episode_length),
                        "loss": float(loss),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_eval_history(path: Path, returns: np.ndarray, lengths: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, (episode_return, episode_length) in enumerate(
            zip(returns, lengths, strict=True)
        ):
            handle.write(
                json.dumps(
                    {
                        "episode": idx,
                        "return": float(episode_return),
                        "length": int(episode_length),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _flatten_params(prefix: str, params: tuple[dict[str, jax.Array], ...]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for layer_idx, layer in enumerate(params):
        for name, value in layer.items():
            arrays[f"{prefix}_layer_{layer_idx}_{name}"] = np.asarray(jax.device_get(value))
    return arrays


def _mean_last(values: np.ndarray, count: int) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values[-count:]))


def _mean_all(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _train_return_metrics(values: np.ndarray) -> dict[str, float | None]:
    return {
        "mean_train_return": _mean_all(values),
        "mean_train_return_last_10": _mean_last(values, 10),
    }


def _resolve_output_path(path: str, run_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    if candidate.is_absolute():
        return candidate
    return (run_dir / candidate).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
