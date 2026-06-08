from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from rlflow_api.services.dataset_analysis import (
    _flatten_observations,
    _train_cfn,
    _train_rnd,
    decode_grid_positions,
)


Algorithm = Literal["rnd", "cfn", "count_oracle", "count_vanilla"]

DEFAULT_DATASET = Path(
    "runs/dataset-generator/dataset-generator-20260527-042201-50c34db4/"
    "dataset_symbolic_2048_distractor_corner.npz"
)
DEFAULT_PLOT_METRICS = (
    "false_known_rate",
    "false_unknown_rate",
    "action_rank_accuracy",
    "distractor_range_mean",
    "oracle_bonus_correlation",
)
ALGORITHM_LABELS = {
    "rnd": "RND",
    "cfn": "CFN",
    "count_oracle": "Count oracle",
    "count_vanilla": "Vanilla count",
}
METRIC_LABELS = {
    "false_known_rate": "False-known rate",
    "false_known_unvisited_rate": "False-known unvisited rate",
    "false_unknown_rate": "False-unknown rate",
    "unvisited_action_recall": "Unvisited action recall",
    "action_rank_accuracy": "Action rank accuracy",
    "action_range_mean": "Action range",
    "distractor_range_mean": "Distractor range",
    "distractor_std_mean": "Distractor std",
    "oracle_bonus_correlation": "Oracle-bonus correlation",
    "raw_bonus_correlation": "Raw-bonus correlation",
}


@dataclass(frozen=True)
class ProbeSet:
    dataset_path: Path
    train_observations: np.ndarray
    train_actions: np.ndarray
    train_cfn_targets: np.ndarray | None
    eval_observations: np.ndarray
    eval_actions: np.ndarray
    items: list[dict[str, Any]]
    action_count: int
    decoding_source: str


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset.expanduser()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset does not exist: {dataset_path}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = _build_probe_set(
        dataset_path,
        count_min_count=args.count_min_count,
        count_bonus_exponent=args.count_bonus_exponent,
        max_probe_observations=args.max_probe_observations,
        probe_seed=args.probe_seed,
    )
    algorithms = _parse_algorithms(args.algorithms)
    seeds = _parse_seeds(args.seeds)
    epoch_sweep = _parse_int_list(args.epoch_sweep, name="--epoch-sweep")
    threshold_sweep = _parse_float_list(args.threshold_sweep, name="--threshold-sweep")
    plot_metrics = _parse_csv_values(args.plot_metrics)
    plot_formats = _parse_csv_values(args.plot_formats)
    bonus_cache: dict[tuple[Algorithm, int, int], tuple[np.ndarray, list[float]]] = {}

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    loss_history: dict[str, dict[str, list[float]]] = defaultdict(dict)

    print(f"dataset: {probe.dataset_path}")
    print(f"probe_items: {len(probe.items)}")
    print(f"state_action_probe_source: {probe.decoding_source}")

    for seed in seeds:
        for algorithm in algorithms:
            print(f"running: {algorithm} seed={seed}")
            bonuses, losses = _cached_algorithm_bonuses(
                bonus_cache,
                algorithm,
                probe,
                args,
                seed=seed,
                epochs=args.epochs,
            )
            if losses:
                loss_history[algorithm][f"seed={seed},epoch={args.epochs}"] = losses
            rows = _rows_for_algorithm(
                probe,
                algorithm,
                seed,
                bonuses,
                epoch=args.epochs,
                threshold=args.rmax_threshold,
            )
            all_rows.extend(rows)
            summary_rows.append(
                _summarize(
                    rows,
                    algorithm=algorithm,
                    seed=seed,
                    epoch=args.epochs,
                    threshold=args.rmax_threshold,
                )
            )

    aggregate_rows = _aggregate_summaries(summary_rows)
    item_csv = out_dir / "bonus_items.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    aggregate_csv = out_dir / "summary_by_algorithm.csv"
    config_json = out_dir / "analysis_config.json"
    loss_json = out_dir / "loss_history.json"
    plot_paths: list[Path] = []

    _write_csv(item_csv, all_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(aggregate_csv, aggregate_rows)

    if epoch_sweep:
        epoch_rows = _run_epoch_sweep(
            probe,
            args,
            algorithms,
            seeds,
            epoch_sweep,
            bonus_cache,
            loss_history,
        )
        epoch_csv = out_dir / "epoch_sweep_summary.csv"
        epoch_aggregate_csv = out_dir / "epoch_sweep_by_algorithm.csv"
        _write_csv(epoch_csv, epoch_rows)
        _write_csv(
            epoch_aggregate_csv,
            _aggregate_summaries(epoch_rows, group_keys=("algorithm", "epoch")),
        )
        plot_paths.extend(
            _plot_sweep(
                epoch_rows,
                x_key="epoch",
                x_label="Epochs",
                out_dir=out_dir / "plots",
                metrics=plot_metrics,
                formats=plot_formats,
                file_prefix="metrics_vs_epochs",
            )
        )
        print(f"epoch_sweep_summary_csv: {epoch_csv}")
        print(f"epoch_sweep_by_algorithm_csv: {epoch_aggregate_csv}")

    if threshold_sweep:
        threshold_rows = _run_threshold_sweep(
            probe,
            args,
            algorithms,
            seeds,
            threshold_sweep,
            bonus_cache,
        )
        threshold_csv = out_dir / "threshold_sweep_summary.csv"
        threshold_aggregate_csv = out_dir / "threshold_sweep_by_algorithm.csv"
        _write_csv(threshold_csv, threshold_rows)
        _write_csv(
            threshold_aggregate_csv,
            _aggregate_summaries(
                threshold_rows,
                group_keys=("algorithm", "rmax_threshold"),
            ),
        )
        plot_paths.extend(
            _plot_sweep(
                threshold_rows,
                x_key="rmax_threshold",
                x_label="R-Max threshold",
                out_dir=out_dir / "plots",
                metrics=plot_metrics,
                formats=plot_formats,
                file_prefix="metrics_vs_rmax_threshold",
            )
        )
        print(f"threshold_sweep_summary_csv: {threshold_csv}")
        print(f"threshold_sweep_by_algorithm_csv: {threshold_aggregate_csv}")

    config_json.write_text(
        json.dumps(
            _config_payload(
                args,
                probe,
                algorithms,
                seeds,
                epoch_sweep=epoch_sweep,
                threshold_sweep=threshold_sweep,
                plot_metrics=plot_metrics,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    loss_json.write_text(json.dumps(loss_history, indent=2) + "\n", encoding="utf-8")

    print(f"bonus_items_csv: {item_csv}")
    print(f"summary_metrics_csv: {summary_csv}")
    print(f"summary_by_algorithm_csv: {aggregate_csv}")
    print(f"analysis_config_json: {config_json}")
    print(f"loss_history_json: {loss_json}")
    for path in plot_paths:
        print(f"plot: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze intrinsic-bonus failure modes on a fixed replay dataset. "
            "The script compares RND, CFN, oracle state-action counts, and exact "
            "raw-observation counts that do not ignore distractors."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Transition .npz dataset. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/analysis/intrinsic_bonus_failures"),
        help="Output directory for CSV/JSON analysis artifacts.",
    )
    parser.add_argument(
        "--algorithms",
        default="rnd,cfn,count_oracle,count_vanilla",
        help=(
            "Comma-separated algorithms to run. Choices: rnd, cfn, "
            "count_oracle, count_vanilla."
        ),
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated learned-method seeds.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-units", default="128,128")
    parser.add_argument(
        "--rnd-activation",
        choices=("relu", "tanh", "gelu", "elu", "linear"),
        default="relu",
    )
    parser.add_argument(
        "--cfn-activation",
        choices=("relu", "tanh", "gelu", "elu", "linear"),
        default="gelu",
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd", "rmsprop"), default="adam")
    parser.add_argument(
        "--rnd-action-conditioning",
        choices=("none", "input", "output", "pair"),
        default="pair",
    )
    parser.add_argument(
        "--cfn-action-conditioning",
        choices=("none", "input", "output", "pair"),
        default="pair",
    )
    parser.add_argument("--update-period", type=int, default=1)
    parser.add_argument("--rnd-output-dim", type=int, default=128)
    parser.add_argument("--cfn-output-dim", type=int, default=128)
    parser.add_argument("--intrinsic-reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--rnd-intrinsic-reward-scale",
        "--rnd-reward-scale",
        dest="rnd_intrinsic_reward_scale",
        type=float,
        default=None,
        help="RND bonus scale. Defaults to --intrinsic-reward-scale when omitted.",
    )
    parser.add_argument(
        "--cfn-intrinsic-reward-scale",
        "--cfn-reward-scale",
        dest="cfn_intrinsic_reward_scale",
        type=float,
        default=None,
        help="CFN bonus scale. Defaults to --intrinsic-reward-scale when omitted.",
    )
    parser.add_argument("--intrinsic-stats-decay", type=float, default=0.9)
    parser.add_argument("--intrinsic-reward-epsilon", type=float, default=1.0)
    parser.add_argument("--intrinsic-reward-clip", type=float, default=100.0)
    parser.add_argument("--intrinsic-reward-center", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--rmax-threshold",
        type=float,
        default=0.5,
        help="R-Max unknown threshold. A method says unknown iff bonus > threshold.",
    )
    parser.add_argument(
        "--epoch-sweep",
        default="",
        help=(
            "Comma-separated epoch counts for metrics-vs-epochs plots. "
            "Example: 1,2,5,10,20,50,100."
        ),
    )
    parser.add_argument(
        "--threshold-sweep",
        default="0.05,0.1,0.2,0.3,0.5,0.7,1.0",
        help=(
            "Comma-separated R-Max thresholds for metrics-vs-threshold plots. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--plot-metrics",
        default=",".join(DEFAULT_PLOT_METRICS),
        help="Comma-separated summary metrics to plot.",
    )
    parser.add_argument(
        "--plot-formats",
        default="png,svg",
        help="Comma-separated plot formats passed to matplotlib savefig.",
    )
    parser.add_argument("--count-min-count", type=float, default=1.0)
    parser.add_argument("--count-bonus-exponent", type=float, default=0.5)
    parser.add_argument(
        "--max-probe-observations",
        type=int,
        default=0,
        help=(
            "Optional cap on unique raw observations used for probing. "
            "0 means use all unique raw observations."
        ),
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=0,
        help="Seed used only when --max-probe-observations samples probe observations.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.update_period < 1:
        parser.error("--update-period must be at least 1")
    if args.rnd_output_dim < 1 or args.cfn_output_dim < 1:
        parser.error("--rnd-output-dim and --cfn-output-dim must be at least 1")
    if args.intrinsic_stats_decay < 0 or args.intrinsic_stats_decay > 1:
        parser.error("--intrinsic-stats-decay must be in [0, 1]")
    if args.intrinsic_reward_scale < 0:
        parser.error("--intrinsic-reward-scale must be non-negative")
    if (
        args.rnd_intrinsic_reward_scale is not None
        and args.rnd_intrinsic_reward_scale < 0
    ):
        parser.error("--rnd-intrinsic-reward-scale must be non-negative")
    if (
        args.cfn_intrinsic_reward_scale is not None
        and args.cfn_intrinsic_reward_scale < 0
    ):
        parser.error("--cfn-intrinsic-reward-scale must be non-negative")
    if args.intrinsic_reward_epsilon <= 0:
        parser.error("--intrinsic-reward-epsilon must be positive")
    if args.intrinsic_reward_clip <= 0:
        parser.error("--intrinsic-reward-clip must be positive")
    if args.rmax_threshold < 0:
        parser.error("--rmax-threshold must be non-negative")
    if args.count_min_count <= 0:
        parser.error("--count-min-count must be positive")
    if args.count_bonus_exponent <= 0:
        parser.error("--count-bonus-exponent must be positive")
    if args.max_probe_observations < 0:
        parser.error("--max-probe-observations must be non-negative")
    return args


def _parse_algorithms(raw: str) -> list[Algorithm]:
    valid = {"rnd", "cfn", "count_oracle", "count_vanilla"}
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - valid)
    if unknown:
        raise SystemExit(f"Unknown algorithm(s): {', '.join(unknown)}")
    if not values:
        raise SystemExit("--algorithms must include at least one algorithm")
    return values  # type: ignore[return-value]


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must include at least one seed")
    return seeds


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    if not raw.strip():
        return []
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if any(value < 1 for value in values):
        raise SystemExit(f"{name} values must be positive integers")
    return sorted(set(values))


def _parse_float_list(raw: str, *, name: str) -> list[float]:
    if not raw.strip():
        return []
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if any(value < 0 for value in values):
        raise SystemExit(f"{name} values must be non-negative")
    return sorted(set(values))


def _parse_csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _build_probe_set(
    dataset_path: Path,
    *,
    count_min_count: float,
    count_bonus_exponent: float,
    max_probe_observations: int,
    probe_seed: int,
) -> ProbeSet:
    data = np.load(dataset_path, allow_pickle=False)
    required = {"observations", "actions", "rewards", "next_observations", "terminals"}
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit(f"Replay dataset is missing arrays: {missing}")

    observations = np.asarray(data["observations"])
    actions = np.asarray(data["actions"], dtype=np.int32).reshape(-1)
    if observations.shape[0] != actions.shape[0]:
        raise SystemExit("observations and actions must have the same first dimension")
    if observations.shape[0] == 0:
        raise SystemExit("Dataset is empty")

    train_observations = _flatten_observations(observations)
    action_count = max(int(np.max(actions, initial=0)) + 1, 4)
    cfn_targets = np.asarray(data["cfn_targets"]) if "cfn_targets" in data.files else None

    unique_obs_indices, unique_obs_counts = _unique_indices_and_counts(train_observations)
    if max_probe_observations and unique_obs_indices.shape[0] > max_probe_observations:
        rng = np.random.default_rng(probe_seed)
        selected = np.sort(
            rng.choice(unique_obs_indices.shape[0], size=max_probe_observations, replace=False)
        )
        unique_obs_indices = unique_obs_indices[selected]
        unique_obs_counts = unique_obs_counts[selected]

    unique_source_observations = observations[unique_obs_indices]
    unique_features = train_observations[unique_obs_indices]
    decoding = decode_grid_positions(observations)
    unique_decoding = decode_grid_positions(unique_source_observations)
    if unique_decoding is not None:
        unique_positions = unique_decoding.positions
        decoding_source = unique_decoding.source
    else:
        unique_positions = np.full((unique_obs_indices.shape[0], 2), -1, dtype=np.int32)
        decoding_source = "raw_observation_fallback"

    if decoding is not None:
        all_positions = decoding.positions.astype(np.int32)
        oracle_state_counts = _count_by_key((tuple(pos) for pos in all_positions))
        oracle_state_action_counts = _count_by_key(
            (tuple((int(pos[0]), int(pos[1]), int(action))) for pos, action in zip(all_positions, actions, strict=True))
        )
    else:
        oracle_state_counts = {}
        oracle_state_action_counts = {}

    raw_obs_keys = [_observation_key(row) for row in train_observations]
    raw_obs_counts = _count_by_key(raw_obs_keys)
    raw_state_action_counts = _count_by_key(
        ((key, int(action)) for key, action in zip(raw_obs_keys, actions, strict=True))
    )

    eval_observations = np.repeat(unique_features, action_count, axis=0)
    eval_actions = np.tile(np.arange(action_count, dtype=np.int32), unique_features.shape[0])
    items: list[dict[str, Any]] = []
    variant_offsets: dict[tuple[int, int], int] = defaultdict(int)

    for raw_obs_order, (source_index, raw_obs_count, position) in enumerate(
        zip(unique_obs_indices, unique_obs_counts, unique_positions, strict=True)
    ):
        raw_key = _observation_key(train_observations[source_index])
        raw_hash = _observation_hash(unique_source_observations[raw_obs_order])
        row = int(position[0])
        col = int(position[1])
        state_key = (row, col)
        if row >= 0 and col >= 0:
            variant_index = variant_offsets[state_key]
            variant_offsets[state_key] += 1
            oracle_state_count = oracle_state_counts.get(state_key, int(raw_obs_count))
        else:
            variant_index = raw_obs_order
            oracle_state_count = int(raw_obs_count)

        for action in range(action_count):
            raw_sa_count = raw_state_action_counts.get((raw_key, action), 0)
            if row >= 0 and col >= 0:
                oracle_sa_count = oracle_state_action_counts.get((row, col, action), 0)
            else:
                oracle_sa_count = raw_sa_count
            items.append(
                {
                    "item_index": len(items),
                    "raw_observation_order": raw_obs_order,
                    "source_index": int(source_index),
                    "raw_observation_hash": raw_hash,
                    "distractor_variant_index": variant_index,
                    "state_row": row if row >= 0 else "",
                    "state_col": col if col >= 0 else "",
                    "action": action,
                    "raw_observation_count": int(raw_obs_count),
                    "raw_state_action_count": int(raw_sa_count),
                    "oracle_state_count": int(oracle_state_count),
                    "oracle_state_action_count": int(oracle_sa_count),
                    "raw_bonus": _count_bonus(
                        raw_sa_count,
                        min_count=count_min_count,
                        exponent=count_bonus_exponent,
                    ),
                    "oracle_bonus": _count_bonus(
                        oracle_sa_count,
                        min_count=count_min_count,
                        exponent=count_bonus_exponent,
                    ),
                }
            )

    return ProbeSet(
        dataset_path=dataset_path,
        train_observations=train_observations,
        train_actions=actions,
        train_cfn_targets=cfn_targets,
        eval_observations=eval_observations,
        eval_actions=eval_actions,
        items=items,
        action_count=action_count,
        decoding_source=decoding_source,
    )


def _unique_indices_and_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _unique_values, unique_indices, unique_counts = np.unique(
        values,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    order = np.argsort(unique_indices)
    return unique_indices[order].astype(np.int32), unique_counts[order].astype(np.int32)


def _count_by_key(keys: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _observation_key(flat_observation: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(flat_observation)
    return contiguous.tobytes()


def _observation_hash(observation: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(observation)
    digest = hashlib.blake2b(contiguous.tobytes(), digest_size=8)
    return digest.hexdigest()


def _count_bonus(count: int | float, *, min_count: float, exponent: float) -> float:
    return float(1.0 / (max(float(count), min_count) ** exponent))


def _cached_algorithm_bonuses(
    cache: dict[tuple[Algorithm, int, int], tuple[np.ndarray, list[float]]],
    algorithm: Algorithm,
    probe: ProbeSet,
    args: argparse.Namespace,
    *,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, list[float]]:
    key = (algorithm, seed, epochs)
    if key not in cache:
        cache[key] = _algorithm_bonuses(
            algorithm,
            probe,
            args,
            seed=seed,
            epochs=epochs,
        )
    return cache[key]


def _algorithm_bonuses(
    algorithm: Algorithm,
    probe: ProbeSet,
    args: argparse.Namespace,
    *,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, list[float]]:
    if algorithm == "count_oracle":
        return np.asarray([item["oracle_bonus"] for item in probe.items], dtype=np.float32), []
    if algorithm == "count_vanilla":
        return np.asarray([item["raw_bonus"] for item in probe.items], dtype=np.float32), []

    hidden_units = _parse_hidden_units(args.hidden_units)
    common = {
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_units": hidden_units,
        "optimizer": args.optimizer,
        "update_period": args.update_period,
        "intrinsic_reward_scale": _intrinsic_reward_scale(algorithm, args),
        "intrinsic_stats_decay": args.intrinsic_stats_decay,
        "intrinsic_reward_epsilon": args.intrinsic_reward_epsilon,
        "intrinsic_reward_clip": args.intrinsic_reward_clip,
        "intrinsic_reward_center": args.intrinsic_reward_center,
        "max_grad_norm": args.max_grad_norm,
        "num_actions": probe.action_count,
        "seed": seed,
    }
    if algorithm == "rnd":
        return _train_rnd(
            probe.train_observations,
            probe.train_actions,
            probe.eval_observations,
            probe.eval_actions,
            activation=args.rnd_activation,
            action_conditioning=args.rnd_action_conditioning,
            output_dim=args.rnd_output_dim,
            **common,
        )
    if algorithm == "cfn":
        return _train_cfn(
            probe.train_observations,
            probe.train_actions,
            probe.eval_observations,
            probe.eval_actions,
            cfn_targets=probe.train_cfn_targets,
            activation=args.cfn_activation,
            action_conditioning=args.cfn_action_conditioning,
            output_dim=args.cfn_output_dim,
            **common,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _intrinsic_reward_scale(algorithm: Algorithm, args: argparse.Namespace) -> float:
    if algorithm == "rnd" and args.rnd_intrinsic_reward_scale is not None:
        return float(args.rnd_intrinsic_reward_scale)
    if algorithm == "cfn" and args.cfn_intrinsic_reward_scale is not None:
        return float(args.cfn_intrinsic_reward_scale)
    return float(args.intrinsic_reward_scale)


def _parse_hidden_units(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _rows_for_algorithm(
    probe: ProbeSet,
    algorithm: Algorithm,
    seed: int,
    bonuses: np.ndarray,
    *,
    epoch: int,
    threshold: float,
) -> list[dict[str, Any]]:
    if bonuses.shape[0] != len(probe.items):
        raise SystemExit(
            f"{algorithm} returned {bonuses.shape[0]} bonuses for {len(probe.items)} probe items"
        )

    rows: list[dict[str, Any]] = []
    for item, bonus in zip(probe.items, bonuses, strict=True):
        oracle_bonus = float(item["oracle_bonus"])
        raw_bonus = float(item["raw_bonus"])
        method_bonus = float(bonus)
        oracle_unknown = oracle_bonus > threshold
        method_unknown = method_bonus > threshold
        rows.append(
            {
                "dataset": str(probe.dataset_path),
                "algorithm": algorithm,
                "seed": seed,
                "epoch": epoch,
                "rmax_threshold": threshold,
                **item,
                "bonus": method_bonus,
                "oracle_bonus": oracle_bonus,
                "raw_bonus": raw_bonus,
                "oracle_unknown": oracle_unknown,
                "method_unknown": method_unknown,
                "false_known": oracle_unknown and not method_unknown,
                "false_unknown": (not oracle_unknown) and method_unknown,
                "observed_oracle_state_action": int(item["oracle_state_action_count"]) > 0,
                "observed_raw_state_action": int(item["raw_state_action_count"]) > 0,
            }
        )
    return rows


def _summarize(
    rows: list[dict[str, Any]],
    *,
    algorithm: Algorithm,
    seed: int,
    epoch: int,
    threshold: float,
) -> dict[str, Any]:
    bonus = _float_array(rows, "bonus")
    oracle_bonus = _float_array(rows, "oracle_bonus")
    raw_bonus = _float_array(rows, "raw_bonus")
    oracle_unknown = _bool_array(rows, "oracle_unknown")
    method_unknown = _bool_array(rows, "method_unknown")
    false_known = _bool_array(rows, "false_known")
    false_unknown = _bool_array(rows, "false_unknown")
    oracle_counts = _float_array(rows, "oracle_state_action_count")
    raw_counts = _float_array(rows, "raw_state_action_count")
    unvisited = oracle_counts <= 0
    rare_observed = np.logical_and(oracle_counts > 0, oracle_unknown)
    known = ~oracle_unknown

    action_metrics = _action_metrics(rows)
    distractor_metrics = _distractor_metrics(rows)

    return {
        "algorithm": algorithm,
        "seed": seed,
        "epoch": epoch,
        "rmax_threshold": threshold,
        "num_probe_items": len(rows),
        "num_oracle_unknown": int(np.sum(oracle_unknown)),
        "num_oracle_known": int(np.sum(~oracle_unknown)),
        "num_method_unknown": int(np.sum(method_unknown)),
        "bonus_mean": _mean(bonus),
        "bonus_median": _median(bonus),
        "bonus_min": _min(bonus),
        "bonus_max": _max(bonus),
        "oracle_bonus_correlation": _pearson(bonus, oracle_bonus),
        "raw_bonus_correlation": _pearson(bonus, raw_bonus),
        "oracle_minus_raw_correlation": _nan_diff(
            _pearson(bonus, oracle_bonus),
            _pearson(bonus, raw_bonus),
        ),
        "false_known_rate": _rate(false_known, oracle_unknown),
        "false_known_unvisited_rate": _rate(false_known, unvisited),
        "false_known_rare_observed_rate": _rate(false_known, rare_observed),
        "false_unknown_rate": _rate(false_unknown, known),
        "unvisited_action_recall": _rate(method_unknown, unvisited),
        "rare_observed_action_recall": _rate(method_unknown, rare_observed),
        "known_action_false_unknown_rate": _rate(method_unknown, known),
        "mean_bonus_oracle_unknown": _masked_mean(bonus, oracle_unknown),
        "mean_bonus_oracle_known": _masked_mean(bonus, known),
        "mean_bonus_unvisited": _masked_mean(bonus, unvisited),
        "mean_bonus_rare_observed": _masked_mean(bonus, rare_observed),
        "mean_bonus_known": _masked_mean(bonus, known),
        "mean_raw_count": _mean(raw_counts),
        "mean_oracle_count": _mean(oracle_counts),
        **action_metrics,
        **distractor_metrics,
    }


def _run_epoch_sweep(
    probe: ProbeSet,
    args: argparse.Namespace,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    epochs: Sequence[int],
    bonus_cache: dict[tuple[Algorithm, int, int], tuple[np.ndarray, list[float]]],
    loss_history: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for epoch in epochs:
        for seed in seeds:
            for algorithm in algorithms:
                print(f"epoch_sweep: {algorithm} seed={seed} epoch={epoch}")
                bonuses, losses = _cached_algorithm_bonuses(
                    bonus_cache,
                    algorithm,
                    probe,
                    args,
                    seed=seed,
                    epochs=epoch,
                )
                if losses:
                    loss_history[algorithm][f"seed={seed},epoch={epoch}"] = losses
                rows = _rows_for_algorithm(
                    probe,
                    algorithm,
                    seed,
                    bonuses,
                    epoch=epoch,
                    threshold=args.rmax_threshold,
                )
                summary_rows.append(
                    _summarize(
                        rows,
                        algorithm=algorithm,
                        seed=seed,
                        epoch=epoch,
                        threshold=args.rmax_threshold,
                    )
                )
    return summary_rows


def _run_threshold_sweep(
    probe: ProbeSet,
    args: argparse.Namespace,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    thresholds: Sequence[float],
    bonus_cache: dict[tuple[Algorithm, int, int], tuple[np.ndarray, list[float]]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for seed in seeds:
            for algorithm in algorithms:
                print(f"threshold_sweep: {algorithm} seed={seed} threshold={threshold}")
                bonuses, _losses = _cached_algorithm_bonuses(
                    bonus_cache,
                    algorithm,
                    probe,
                    args,
                    seed=seed,
                    epochs=args.epochs,
                )
                rows = _rows_for_algorithm(
                    probe,
                    algorithm,
                    seed,
                    bonuses,
                    epoch=args.epochs,
                    threshold=threshold,
                )
                summary_rows.append(
                    _summarize(
                        rows,
                        algorithm=algorithm,
                        seed=seed,
                        epoch=args.epochs,
                        threshold=threshold,
                    )
                )
    return summary_rows


def _action_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    oracle_counts: dict[tuple[int, int, int], float] = {}
    for row in rows:
        state_row = row["state_row"]
        state_col = row["state_col"]
        if state_row == "" or state_col == "":
            continue
        key = (int(state_row), int(state_col), int(row["action"]))
        grouped[key].append(float(row["bonus"]))
        oracle_counts[key] = float(row["oracle_state_action_count"])

    by_state: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    by_state_count: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for (state_row, state_col, action), values in grouped.items():
        state_key = (state_row, state_col)
        by_state[state_key][action] = float(np.mean(values))
        by_state_count[state_key][action] = oracle_counts[(state_row, state_col, action)]

    ranges: list[float] = []
    stds: list[float] = []
    rank_scores: list[float] = []
    for state_key, action_values in by_state.items():
        if len(action_values) < 2:
            continue
        values = np.asarray(list(action_values.values()), dtype=np.float64)
        ranges.append(float(np.max(values) - np.min(values)))
        stds.append(float(np.std(values)))
        counts = by_state_count[state_key]
        actions = sorted(action_values)
        for index, action_a in enumerate(actions):
            for action_b in actions[index + 1 :]:
                count_a = counts[action_a]
                count_b = counts[action_b]
                if count_a == count_b:
                    continue
                bonus_a = action_values[action_a]
                bonus_b = action_values[action_b]
                if bonus_a == bonus_b:
                    rank_scores.append(0.5)
                else:
                    rank_scores.append(
                        1.0
                        if (count_a < count_b and bonus_a > bonus_b)
                        or (count_b < count_a and bonus_b > bonus_a)
                        else 0.0
                    )

    return {
        "action_range_mean": _mean_list(ranges),
        "action_range_median": _median_list(ranges),
        "action_range_max": _max_list(ranges),
        "action_std_mean": _mean_list(stds),
        "action_rank_accuracy": _mean_list(rank_scores),
        "action_rank_pair_count": len(rank_scores),
    }


def _distractor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        state_row = row["state_row"]
        state_col = row["state_col"]
        if state_row == "" or state_col == "":
            continue
        key = (int(state_row), int(state_col), int(row["action"]))
        grouped[key].append(float(row["bonus"]))

    ranges: list[float] = []
    stds: list[float] = []
    for values in grouped.values():
        if len(values) < 2:
            continue
        array = np.asarray(values, dtype=np.float64)
        ranges.append(float(np.max(array) - np.min(array)))
        stds.append(float(np.std(array)))

    return {
        "distractor_group_count": len(ranges),
        "distractor_range_mean": _mean_list(ranges),
        "distractor_range_median": _median_list(ranges),
        "distractor_range_max": _max_list(ranges),
        "distractor_std_mean": _mean_list(stds),
        "distractor_std_median": _median_list(stds),
    }


def _aggregate_summaries(
    summary_rows: list[dict[str, Any]],
    *,
    group_keys: Sequence[str] = ("algorithm",),
) -> list[dict[str, Any]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_group[tuple(row[key] for key in group_keys)].append(row)

    output: list[dict[str, Any]] = []
    excluded = {"algorithm", "seed", "epoch", "rmax_threshold", *group_keys}
    for group, rows in sorted(by_group.items(), key=lambda item: item[0]):
        aggregate: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        aggregate["seed_count"] = len(rows)
        metric_names = [
            key
            for key, value in rows[0].items()
            if key not in excluded and isinstance(value, int | float)
        ]
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric_name}_mean"] = float(np.mean(finite)) if finite.size else math.nan
            aggregate[f"{metric_name}_std"] = float(np.std(finite)) if finite.size else math.nan
        output.append(aggregate)
    return output


def _float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _bool_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([bool(row[key]) for row in rows], dtype=bool)


def _rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = int(np.sum(denominator_mask))
    if denominator == 0:
        return math.nan
    return float(np.sum(np.logical_and(numerator_mask, denominator_mask)) / denominator)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return float(np.mean(values[mask]))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.logical_and(np.isfinite(left), np.isfinite(right))
    if int(np.sum(mask)) < 2:
        return math.nan
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _nan_diff(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan
    return left - right


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else math.nan


def _min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else math.nan


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def _mean_list(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _median_list(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else math.nan


def _max_list(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_sweep(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    out_dir: Path,
    metrics: Sequence[str],
    formats: Sequence[str],
    file_prefix: str,
) -> list[Path]:
    if not rows or not metrics:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib could not be imported: {exc}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    available_metrics = [metric for metric in metrics if metric in rows[0]]
    missing_metrics = sorted(set(metrics) - set(available_metrics))
    if missing_metrics:
        print(f"Skipping unknown plot metric(s): {', '.join(missing_metrics)}")
    if not available_metrics:
        return []

    paths: list[Path] = []
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        _plot_metric_lines(ax, rows, x_key=x_key, metric=metric)
        ax.set_xlabel(x_label)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs {x_label}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _maybe_log_epoch_axis(ax, rows, x_key)
        paths.extend(_save_figure(fig, out_dir / f"{file_prefix}_{_slug(metric)}", formats))
        plt.close(fig)

    grid_metrics = available_metrics[:6]
    if len(grid_metrics) > 1:
        col_count = 2
        row_count = math.ceil(len(grid_metrics) / col_count)
        fig, axes = plt.subplots(
            row_count,
            col_count,
            figsize=(7.2 * col_count, 3.8 * row_count),
            squeeze=False,
        )
        for index, metric in enumerate(grid_metrics):
            ax = axes[index // col_count][index % col_count]
            _plot_metric_lines(ax, rows, x_key=x_key, metric=metric)
            ax.set_xlabel(x_label)
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_title(METRIC_LABELS.get(metric, metric))
            ax.grid(True, alpha=0.25)
            _maybe_log_epoch_axis(ax, rows, x_key)
            if index == 0:
                ax.legend(frameon=False)
        for index in range(len(grid_metrics), row_count * col_count):
            axes[index // col_count][index % col_count].axis("off")
        fig.suptitle(f"Intrinsic bonus metrics vs {x_label}", y=0.995)
        fig.tight_layout()
        paths.extend(_save_figure(fig, out_dir / file_prefix, formats))
        plt.close(fig)
    return paths


def _plot_metric_lines(ax: Any, rows: list[dict[str, Any]], *, x_key: str, metric: str) -> None:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row.get(metric, math.nan))
        x_value = float(row[x_key])
        if math.isfinite(value) and math.isfinite(x_value):
            grouped[str(row["algorithm"])][x_value].append(value)

    for algorithm in sorted(grouped, key=_algorithm_sort_key):
        x_values = sorted(grouped[algorithm])
        means: list[float] = []
        stds: list[float] = []
        for x_value in x_values:
            values = np.asarray(grouped[algorithm][x_value], dtype=np.float64)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        label = ALGORITHM_LABELS.get(algorithm, algorithm)
        ax.plot(x_values, means, marker="o", linewidth=1.8, markersize=4.5, label=label)
        if any(std > 0.0 for std in stds):
            x_array = np.asarray(x_values, dtype=np.float64)
            mean_array = np.asarray(means, dtype=np.float64)
            std_array = np.asarray(stds, dtype=np.float64)
            ax.fill_between(
                x_array,
                mean_array - std_array,
                mean_array + std_array,
                alpha=0.12,
            )


def _algorithm_sort_key(algorithm: str) -> int:
    order = {"rnd": 0, "cfn": 1, "count_oracle": 2, "count_vanilla": 3}
    return order.get(algorithm, len(order))


def _maybe_log_epoch_axis(ax: Any, rows: list[dict[str, Any]], x_key: str) -> None:
    if x_key != "epoch":
        return
    x_values = [float(row[x_key]) for row in rows if float(row[x_key]) > 0.0]
    if not x_values:
        return
    if max(x_values) / min(x_values) >= 10.0:
        ax.set_xscale("log")


def _save_figure(fig: Any, path_base: Path, formats: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        path = path_base.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(path)
    return paths


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _config_payload(
    args: argparse.Namespace,
    probe: ProbeSet,
    algorithms: Sequence[Algorithm],
    seeds: Sequence[int],
    *,
    epoch_sweep: Sequence[int],
    threshold_sweep: Sequence[float],
    plot_metrics: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": str(probe.dataset_path),
        "algorithms": list(algorithms),
        "seeds": list(seeds),
        "probe_items": len(probe.items),
        "action_count": probe.action_count,
        "decoding_source": probe.decoding_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_units": _parse_hidden_units(args.hidden_units),
        "rnd_activation": args.rnd_activation,
        "cfn_activation": args.cfn_activation,
        "optimizer": args.optimizer,
        "rnd_action_conditioning": args.rnd_action_conditioning,
        "cfn_action_conditioning": args.cfn_action_conditioning,
        "update_period": args.update_period,
        "rnd_output_dim": args.rnd_output_dim,
        "cfn_output_dim": args.cfn_output_dim,
        "intrinsic_reward_scale": args.intrinsic_reward_scale,
        "rnd_intrinsic_reward_scale": (
            args.rnd_intrinsic_reward_scale
            if args.rnd_intrinsic_reward_scale is not None
            else args.intrinsic_reward_scale
        ),
        "cfn_intrinsic_reward_scale": (
            args.cfn_intrinsic_reward_scale
            if args.cfn_intrinsic_reward_scale is not None
            else args.intrinsic_reward_scale
        ),
        "intrinsic_stats_decay": args.intrinsic_stats_decay,
        "intrinsic_reward_epsilon": args.intrinsic_reward_epsilon,
        "intrinsic_reward_clip": args.intrinsic_reward_clip,
        "intrinsic_reward_center": args.intrinsic_reward_center,
        "max_grad_norm": args.max_grad_norm,
        "rmax_threshold": args.rmax_threshold,
        "epoch_sweep": list(epoch_sweep),
        "threshold_sweep": list(threshold_sweep),
        "plot_metrics": list(plot_metrics),
        "plot_formats": _parse_csv_values(args.plot_formats),
        "count_min_count": args.count_min_count,
        "count_bonus_exponent": args.count_bonus_exponent,
        "max_probe_observations": args.max_probe_observations,
        "probe_seed": args.probe_seed,
    }


if __name__ == "__main__":
    main()
