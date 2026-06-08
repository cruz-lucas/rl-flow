from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from rlflow.analysis.curves import interpolate_and_aggregate, smooth_curve_columns
from rlflow.analysis.loading import load_histories
from rlflow.analysis.plotting import plot_learning_curves


def main() -> None:
    args = _parse_args()
    sweep_specs = [_parse_sweep_spec(raw) for raw in args.sweeps]
    labels = _resolve_labels(sweep_specs, args.label)

    frames: list[pd.DataFrame] = []
    for (provided_label, sweep_path), label in zip(sweep_specs, labels, strict=True):
        del provided_label
        history = load_histories(sweep_path, history=args.history)
        if history.empty:
            raise SystemExit(f"No {args.history} histories found for sweep: {sweep_path}")
        history = history.copy()
        history["source_sweep"] = str(sweep_path)
        history["comparison_label"] = label
        history["group_id"] = label
        history["parameters"] = [{"sweep": label} for _ in range(len(history))]
        frames.append(history)

    raw = pd.concat(frames, ignore_index=True)
    dropped_partial_count = 0
    if args.drop_final_partial:
        raw, dropped_partial_count = _drop_final_partial_episodes(
            raw,
            max_episode_steps=args.max_episode_steps,
        )

    curves = interpolate_and_aggregate(
        raw,
        x=args.x,
        y=args.y,
        points=args.points,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if curves.empty:
        raise SystemExit("No curve data found after interpolation")

    curves["label"] = curves["group_id"]
    plot_curves = smooth_curve_columns(curves, window=args.smooth_window)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "curves_raw.csv"
    curves_csv = out_dir / "curves.csv"
    raw[_raw_export_columns(raw, args.x, args.y)].to_csv(raw_csv, index=False)
    curves.to_csv(curves_csv, index=False)

    plot_curves_csv: Path | None = None
    if args.smooth_window and args.smooth_window > 1:
        plot_curves_csv = out_dir / "curves_plot.csv"
        plot_curves.to_csv(plot_curves_csv, index=False)

    plot_paths = plot_learning_curves(
        plot_curves,
        out_dir=out_dir,
        x_label=args.x_label,
        y_label=args.y_label,
        title=args.title,
        legend_title=args.legend_title,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )

    print(f"raw_csv: {raw_csv}")
    print(f"curves_csv: {curves_csv}")
    if args.drop_final_partial:
        print(f"dropped_final_partial_episodes: {dropped_partial_count}")
    if plot_curves_csv is not None:
        print(f"curves_plot_csv: {plot_curves_csv}")
    for fmt, path in plot_paths.items():
        print(f"{fmt}: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare seed-averaged learning curves from multiple sweep directories. "
            "Each sweep is treated as one plotted curve."
        )
    )
    parser.add_argument(
        "sweeps",
        nargs="+",
        help="Sweep directory or manifest path. Use Label=path to set a curve label.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Curve label. Repeat once per sweep. Ignored for sweeps passed as Label=path.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/analysis/sweep_discounted_return_compare"))
    parser.add_argument("--history", choices=("train", "eval"), default="train")
    parser.add_argument("--x", default="env_step")
    parser.add_argument("--y", "--value", dest="y", default="discounted_return")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument(
        "--drop-final-partial",
        action="store_true",
        help=(
            "Drop the last logged row for each trial when its episode length is "
            "shorter than --max-episode-steps."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Episode step cap used by --drop-final-partial.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--legend-title", default=None)
    parser.add_argument("--x-label", default="Environment steps")
    parser.add_argument("--y-label", default="Discounted return")
    parser.add_argument("--width", type=float, default=9)
    parser.add_argument("--height", type=float, default=6)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if args.smooth_window is not None and args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be at least 1")
    if args.drop_final_partial and args.max_episode_steps is None:
        parser.error("--drop-final-partial requires --max-episode-steps")
    return args


def _parse_sweep_spec(raw: str) -> tuple[str | None, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if label:
            return label, Path(path).expanduser()
    return None, Path(raw).expanduser()


def _resolve_labels(
    sweep_specs: Sequence[tuple[str | None, Path]],
    cli_labels: Sequence[str],
) -> list[str]:
    unlabeled_count = sum(1 for label, _path in sweep_specs if label is None)
    if cli_labels and len(cli_labels) != unlabeled_count:
        raise SystemExit(
            f"Expected {unlabeled_count} --label values for unlabeled sweeps, got {len(cli_labels)}"
        )

    labels: list[str] = []
    label_iter = iter(cli_labels)
    for label, path in sweep_specs:
        labels.append(label or next(label_iter, _default_label(path)))
    return _dedupe_labels(labels)


def _raw_export_columns(raw: pd.DataFrame, x: str, y: str) -> list[str]:
    candidates = [
        "comparison_label",
        "source_sweep",
        "trial_id",
        "seed_value",
        "episode",
        "env_step",
        "return",
        "discounted_return",
        "length",
        "loss",
        x,
        y,
    ]
    columns: list[str] = []
    for column in candidates:
        if column in raw.columns and column not in columns:
            columns.append(column)
    return columns


def _drop_final_partial_episodes(
    raw: pd.DataFrame,
    *,
    max_episode_steps: int,
) -> tuple[pd.DataFrame, int]:
    if "length" not in raw.columns:
        raise SystemExit("--drop-final-partial requires a length column in histories")

    trial_columns = [
        column
        for column in ("comparison_label", "source_sweep", "trial_id")
        if column in raw.columns
    ]
    if not trial_columns:
        raise SystemExit("--drop-final-partial requires trial_id or source sweep columns")

    order_column = "env_step" if "env_step" in raw.columns else "episode"
    if order_column not in raw.columns:
        raise SystemExit("--drop-final-partial requires env_step or episode in histories")

    drop_indices: list[int] = []
    for _trial_key, trial in raw.groupby(trial_columns, sort=False, dropna=False):
        ordered = trial.copy()
        ordered[order_column] = pd.to_numeric(ordered[order_column], errors="coerce")
        ordered = ordered.sort_values(order_column, kind="stable")
        if ordered.empty:
            continue
        last = ordered.iloc[-1]
        length = pd.to_numeric(pd.Series([last["length"]]), errors="coerce").iloc[0]
        if pd.notna(length) and float(length) < float(max_episode_steps):
            drop_indices.append(int(last.name))

    if not drop_indices:
        return raw, 0
    return raw.drop(index=drop_indices).reset_index(drop=True), len(drop_indices)


def _default_label(path: Path) -> str:
    candidate = path.parent.name if path.name == "sweep_manifest.yaml" else path.name
    return candidate.removesuffix("--best")


def _dedupe_labels(labels: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for label in labels:
        count = counts.get(label, 0)
        counts[label] = count + 1
        deduped.append(label if count == 0 else f"{label}-{count + 1}")
    return deduped


if __name__ == "__main__":
    main()
