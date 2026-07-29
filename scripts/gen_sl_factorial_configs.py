"""Generate the Strehl & Littman factorial + optimistic-exploration configs.

This is the single source of truth for the RiverSwim / SixArms study. Running it
emits (under ``configs/``) the 16 factorial cell workflows + seed sweeps per env,
the 4 comparison-algorithm workflows + sweeps per env, and a factorial design CSV
for ``scripts/analyze_factorial_design.py``.

The 2^4 factorial over tabular Q-learning has four binary factors:

    A epsilon_greedy   0 -> greedy (epsilon=0)      1 -> epsilon-greedy
    B optimistic_init  0 -> Q0 = 0                   1 -> Q0 = V_max
    C count_bonus      0 -> off (beta=0)             1 -> count-based intrinsic reward
    D replay           0 -> no replay buffer         1 -> 5k buffer, replay to convergence

experiment_number = index + 1, with (A, B, C, D) = the little-endian bits of ``index``
(A cycles fastest) -- standard Yates order.

HYPERPARAMETERS ARE EDITABLE BELOW. The reward-scale-dependent bonuses (count_beta,
mbie_beta) and the R-Max known-count threshold ``m`` are literature-informed starting
points for these exact reward scales -- review/tune before the production run.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

NUM_SEEDS = 100
HORIZON = 5000  # env steps per run (single continuing episode); Strehl & Littman setup
DISCOUNT = 0.95
PLANNING_ITERATIONS = 500  # value-iteration sweeps for R-Max / MBIE-EB at gamma=0.95

FACTORS = ("epsilon_greedy", "optimistic_init", "count_bonus", "replay")

# ---------------------------------------------------------------------------
# Per-environment hyperparameters (EDIT HERE).
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    "riverswim": {
        "env_component": "builtin.env.riverswim",
        "env_config": {
            "num_states": 6,
            "start_state": 0,
            # Strehl & Littman: start 50/50 in the two leftmost states (0 and 1).
            "random_start": True,
            "p_left": 0.1,
            "p_stay": 0.6,
            "p_right": 0.3,
            "easy_reward": 5.0,
            "hard_reward": 10000.0,
            "common_reward": 0.0,
        },
        "r_max": 10000.0,
        # V_max = R_max / (1 - gamma) = 10000 / 0.05
        "v_max": 200000.0,
        "q_learning_alpha": 0.1,  # constant step size (tune)
        "epsilon": 0.1,
        "count_beta": 10000.0,  # PLACEHOLDER ~ R_max (tune)
        "mbie_beta": 10000.0,  # PLACEHOLDER ~ R_max (tune)
        "rmax_m": 5,  # R-Max known-count threshold (tune)
        "replay_alpha": 0.1,  # step size for the replay-based agents
    },
    "sixarms": {
        "env_component": "builtin.env.sixarms",
        "env_config": {
            "success_probabilities": [1.0, 0.15, 0.1, 0.05, 0.03, 0.01],
            "arm_rewards": [0.0, 50.0, 133.0, 300.0, 800.0, 1660.0, 6000.0],
        },
        "r_max": 6000.0,
        # V_max = R_max / (1 - gamma) = 6000 / 0.05
        "v_max": 120000.0,
        "q_learning_alpha": 0.1,
        "epsilon": 0.1,
        "count_beta": 6000.0,  # PLACEHOLDER ~ R_max (tune)
        "mbie_beta": 6000.0,  # PLACEHOLDER ~ R_max (tune)
        "rmax_m": 5,
        "replay_alpha": 0.1,
    },
}

# Replay buffer used by the "replay on" factorial cells and the replay-based agents:
# holds every sample (5k = the full horizon) and replays each step to convergence.
REPLAY_CONFIG = {
    "capacity": HORIZON,
    "batch_size": 32,
    "min_size": 1,
    "updates_per_step": 1,
    "replay_until_convergence": True,
    "convergence_tol": 1.0,  # absolute max|dQ| stopping threshold (tune to reward scale)
    "max_replay_iters": 100,  # hard cap on replay passes per env step
}


def _runner_node(node_id: str = "runner") -> dict:
    return {
        "id": node_id,
        "component": "builtin.runner.tabular_jax",
        "position": {"x": 480, "y": 160},
        "config": {
            "seed": 0,
            "train_episodes": 1,
            "train_steps": HORIZON,
            "max_episode_steps": HORIZON,
            "eval_episodes": 0,
            "checkpoint_freq": None,
            "checkpoint_dir": "checkpoints",
            "save_final_checkpoint": False,
        },
    }


def _env_node(env: dict) -> dict:
    return {
        "id": "env",
        "component": env["env_component"],
        "position": {"x": 120, "y": 160},
        "config": dict(env["env_config"]),
    }


def _replay_node() -> dict:
    return {
        "id": "replay",
        "component": "builtin.replay.tabular_uniform",
        "position": {"x": 300, "y": 320},
        "config": dict(REPLAY_CONFIG),
    }


def _workflow(name: str, description: str, experiment_id: str, nodes: list, edges: list) -> dict:
    return {
        "name": name,
        "description": description,
        "execution": {"backend": "local", "cluster": None, "options": {}},
        "metadata": {"experiment_id": experiment_id},
        "nodes": nodes,
        "edges": edges,
    }


def _factorial_workflow(env_name: str, env: dict, index: int) -> dict:
    levels = {factor: (index >> bit) & 1 for bit, factor in enumerate(FACTORS)}
    number = index + 1

    agent = {
        "id": "agent",
        "component": "builtin.agent.q_learning_tabular",
        "position": {"x": 120, "y": 20},
        "config": {
            "learning_rate": env["q_learning_alpha"],
            "discount": DISCOUNT,
            "initial_q": env["v_max"] if levels["optimistic_init"] else 0.0,
            "count_bonus_beta": env["count_beta"] if levels["count_bonus"] else 0.0,
        },
    }
    policy = {
        "id": "policy",
        "component": "builtin.policy.epsilon_greedy",
        "position": {"x": 120, "y": 300},
        "config": {
            "epsilon": env["epsilon"] if levels["epsilon_greedy"] else 0.0,
            "eval_epsilon": 0.0,
        },
    }
    nodes = [_env_node(env), agent, policy, _runner_node()]
    edges = [
        {
            "from_node": "env",
            "from_port": "environment",
            "to_node": "runner",
            "to_port": "environment",
        },
        {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
        {"from_node": "policy", "from_port": "policy", "to_node": "runner", "to_port": "policy"},
    ]
    if levels["replay"]:
        nodes.insert(3, _replay_node())
        edges.append(
            {
                "from_node": "replay",
                "from_port": "replay_buffer",
                "to_node": "runner",
                "to_port": "replay_buffer",
            }
        )

    tags = (
        "".join(letter for letter, factor in zip("EOCR", FACTORS, strict=True) if levels[factor])
        or "base"
    )
    return _workflow(
        name=f"sl_{env_name}_factorial__experiment{number}",
        description=(
            f"S&L {env_name} tabular Q-learning factorial cell {number}/16 [{tags}]: "
            + ", ".join(f"{factor}={levels[factor]}" for factor in FACTORS)
        ),
        experiment_id=f"sl-{env_name}-factorial--experiment{number}",
        nodes=nodes,
        edges=edges,
    )


def _comparison_workflows(env_name: str, env: dict) -> dict[str, dict]:
    base_edges = [
        {
            "from_node": "env",
            "from_port": "environment",
            "to_node": "runner",
            "to_port": "environment",
        },
        {"from_node": "agent", "from_port": "agent", "to_node": "runner", "to_port": "agent"},
    ]
    replay_edge = {
        "from_node": "replay",
        "from_port": "replay_buffer",
        "to_node": "runner",
        "to_port": "replay_buffer",
    }

    def agent_node(component: str, config: dict) -> dict:
        return {
            "id": "agent",
            "component": component,
            "position": {"x": 120, "y": 20},
            "config": config,
        }

    rmax = _workflow(
        name=f"sl_{env_name}_compare__rmax",
        description=f"S&L {env_name} model-based R-Max baseline.",
        experiment_id=f"sl-{env_name}-compare--rmax",
        nodes=[
            _env_node(env),
            agent_node(
                "builtin.agent.rmax_tabular",
                {
                    "discount": DISCOUNT,
                    "known_count_threshold": env["rmax_m"],
                    "rmax_v_max": env["v_max"],
                    "planning_iterations": PLANNING_ITERATIONS,
                },
            ),
            _runner_node(),
        ],
        edges=list(base_edges),
    )
    mbie = _workflow(
        name=f"sl_{env_name}_compare__mbie_eb",
        description=f"S&L {env_name} model-based MBIE-EB baseline.",
        experiment_id=f"sl-{env_name}-compare--mbie_eb",
        nodes=[
            _env_node(env),
            agent_node(
                "builtin.agent.mbie_eb_tabular",
                {
                    "discount": DISCOUNT,
                    "mbie_beta": env["mbie_beta"],
                    "rmax_v_max": env["v_max"],
                    "planning_iterations": PLANNING_ITERATIONS,
                },
            ),
            _runner_node(),
        ],
        edges=list(base_edges),
    )
    replay_rmax = _workflow(
        name=f"sl_{env_name}_compare__replay_rmax",
        description=f"S&L {env_name} replay-based optimistic Q-learning (R-Max variant).",
        experiment_id=f"sl-{env_name}-compare--replay_rmax",
        nodes=[
            _env_node(env),
            agent_node(
                "builtin.agent.replay_rmax_tabular",
                {
                    "learning_rate": env["replay_alpha"],
                    "discount": DISCOUNT,
                    "known_count_threshold": env["rmax_m"],
                    "rmax_v_max": env["v_max"],
                },
            ),
            _replay_node(),
            _runner_node(),
        ],
        edges=[*base_edges, replay_edge],
    )
    replay_mbie = _workflow(
        name=f"sl_{env_name}_compare__replay_mbie_eb",
        description=f"S&L {env_name} replay-based optimistic Q-learning (MBIE-EB variant).",
        experiment_id=f"sl-{env_name}-compare--replay_mbie_eb",
        nodes=[
            _env_node(env),
            agent_node(
                "builtin.agent.replay_mbie_eb_tabular",
                {
                    "learning_rate": env["replay_alpha"],
                    "discount": DISCOUNT,
                    "mbie_beta": env["mbie_beta"],
                    "rmax_v_max": env["v_max"],
                },
            ),
            _replay_node(),
            _runner_node(),
        ],
        edges=[*base_edges, replay_edge],
    )
    return {
        "rmax": rmax,
        "mbie_eb": mbie,
        "replay_rmax": replay_rmax,
        "replay_mbie_eb": replay_mbie,
    }


def _sweep(name: str, sweep_id: str, description: str, workflow_rel: str) -> dict:
    return {
        "name": name,
        "description": description,
        "sweep_id": sweep_id,
        "workflow": workflow_rel,
        "method": "grid",
        "metric": {"goal": "maximize", "name": "cumulative_train_reward"},
        "parameters": {
            "seed": {
                "target": "nodes.runner.config.seed",
                "values": list(range(NUM_SEEDS)),
            }
        },
    }


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def _rel(workflow_path: Path, sweep_path: Path) -> str:
    return os.path.relpath(workflow_path, start=sweep_path.parent)


def main() -> int:
    written = 0
    for env_name, env in ENVIRONMENTS.items():
        wf_dir = CONFIGS / "workflows" / "sl" / env_name
        sw_dir = CONFIGS / "sweeps" / "sl" / env_name

        # --- factorial cells -------------------------------------------------
        design_rows = []
        for index in range(16):
            number = index + 1
            wf = _factorial_workflow(env_name, env, index)
            wf_path = wf_dir / "factorial" / f"experiment{number}.yaml"
            _dump_yaml(wf_path, wf)
            written += 1

            sw_path = sw_dir / "factorial" / f"experiment{number}_{NUM_SEEDS}_seeds.yaml"
            sweep = _sweep(
                name=f"sl_{env_name}_factorial__experiment{number}_{NUM_SEEDS}_seeds",
                sweep_id=f"sl-{env_name}-factorial-experiment{number}-{NUM_SEEDS}-seeds",
                description=f"{NUM_SEEDS} seed replicates for {env_name} factorial cell {number}.",
                workflow_rel=_rel(wf_path, sw_path),
            )
            _dump_yaml(sw_path, sweep)
            written += 1

            levels = {factor: (index >> bit) & 1 for bit, factor in enumerate(FACTORS)}
            design_rows.append({"experiment_number": number, **levels})

        design_path = sw_dir / "factorial_design.csv"
        design_path.parent.mkdir(parents=True, exist_ok=True)
        with design_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["experiment_number", *FACTORS])
            writer.writeheader()
            writer.writerows(design_rows)
        written += 1

        # --- comparison algorithms ------------------------------------------
        for key, wf in _comparison_workflows(env_name, env).items():
            wf_path = wf_dir / "compare" / f"{key}.yaml"
            _dump_yaml(wf_path, wf)
            written += 1

            sw_path = sw_dir / "compare" / f"{key}_{NUM_SEEDS}_seeds.yaml"
            sweep = _sweep(
                name=f"sl_{env_name}_compare__{key}_{NUM_SEEDS}_seeds",
                sweep_id=f"sl-{env_name}-compare-{key}-{NUM_SEEDS}-seeds",
                description=f"{NUM_SEEDS} seed replicates for {env_name} comparison agent {key}.",
                workflow_rel=_rel(wf_path, sw_path),
            )
            _dump_yaml(sw_path, sweep)
            written += 1

    print(f"Wrote {written} files under {CONFIGS}/(workflows|sweeps)/sl/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
