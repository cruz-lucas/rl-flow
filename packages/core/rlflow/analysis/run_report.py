"""Single-run analysis: summarize and plot one run directory.

The sweep tooling all keys off a ``sweep_manifest.yaml``; this module gives the
same first-class treatment to a single run produced by ``rlflow run`` so a
researcher iterating on one config can quickly inspect status, metrics, and the
learning curve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rlflow.analysis.curves import interpolate_and_aggregate
from rlflow.analysis.plotting import plot_learning_curves

_HISTORY_FILES = {"train": "train_history.jsonl", "eval": "eval_history.jsonl"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_run_history(run_dir: str | Path, history: str = "train") -> pd.DataFrame:
    """Load a single run's ``logs/<history>_history.jsonl`` into a DataFrame."""
    if history not in _HISTORY_FILES:
        raise ValueError("history must be 'train' or 'eval'")
    path = Path(run_dir) / "logs" / _HISTORY_FILES[history]
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def _column_stats(df: pd.DataFrame, column: str) -> dict[str, float] | None:
    if df.empty or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return {
        "final": float(series.iloc[-1]),
        "mean": float(series.mean()),
        "max": float(series.max()),
    }


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Collect status, provenance, metrics, and return statistics for one run."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "summaries" / "metrics.json") or _read_json(
        run_dir / "metrics.json"
    )
    train = load_run_history(run_dir, "train")
    eval_history = load_run_history(run_dir, "eval")

    total_env_steps: int | None = None
    if not train.empty and "env_step" in train.columns:
        steps = pd.to_numeric(train["env_step"], errors="coerce").dropna()
        total_env_steps = int(steps.max()) if not steps.empty else None

    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "status": status.get("status"),
        "backend": manifest.get("backend"),
        "seed": manifest.get("seed"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "metrics": metrics,
        "train_episodes": int(len(train)),
        "eval_episodes": int(len(eval_history)),
        "total_env_steps": total_env_steps,
        "train_return": _column_stats(train, "return"),
        "train_discounted_return": _column_stats(train, "discounted_return"),
        "eval_return": _column_stats(eval_history, "return"),
        "eval_discounted_return": _column_stats(eval_history, "discounted_return"),
    }


def format_run_report(summary: dict[str, Any]) -> str:
    """Render :func:`summarize_run` output as a human-readable report."""
    lines = [
        f"Run:      {summary.get('run_id') or Path(summary['run_dir']).name}",
        f"Dir:      {summary['run_dir']}",
        f"Status:   {summary.get('status')}",
        f"Backend:  {summary.get('backend')}    Seed: {summary.get('seed')}",
    ]
    commit = summary.get("git_commit")
    if commit:
        dirty = " (dirty)" if summary.get("git_dirty") else ""
        lines.append(f"Commit:   {commit}{dirty}")

    lines.append("")
    lines.append(
        f"Train episodes: {summary['train_episodes']}    Eval episodes: {summary['eval_episodes']}"
    )
    if summary.get("total_env_steps") is not None:
        lines.append(f"Total env steps: {summary['total_env_steps']}")

    def _append_stats(label: str, stats: dict[str, float] | None) -> None:
        if stats is not None:
            lines.append(
                f"  {label:<26} final={stats['final']:.4g}  "
                f"mean={stats['mean']:.4g}  max={stats['max']:.4g}"
            )

    lines.append("")
    lines.append("Returns:")
    _append_stats("train return", summary["train_return"])
    _append_stats("train discounted_return", summary["train_discounted_return"])
    _append_stats("eval return", summary["eval_return"])
    _append_stats("eval discounted_return", summary["eval_discounted_return"])

    metrics = summary.get("metrics") or {}
    scalar = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if scalar:
        lines.append("")
        lines.append("Summary metrics:")
        for key in sorted(scalar):
            lines.append(f"  {key}: {scalar[key]}")

    return "\n".join(lines)


def plot_run_curve(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    history: str = "train",
) -> Path | None:
    """Write a learning-curve PNG for a single run; returns the path or None."""
    run_dir = Path(run_dir)
    history_df = load_run_history(run_dir, history)
    if history_df.empty:
        return None

    x = "env_step" if "env_step" in history_df.columns else "episode"
    has_discounted = "discounted_return" in history_df.columns
    y = "discounted_return" if has_discounted else "return"
    if x not in history_df.columns or (
        y not in history_df.columns and "return" not in history_df.columns
    ):
        return None

    prepared = history_df.copy()
    prepared["trial_id"] = run_dir.name
    prepared["parameters"] = [{} for _ in range(len(prepared))]
    prepared["seed_value"] = 0

    curves = interpolate_and_aggregate(prepared, x=x, y=y)
    if curves.empty:
        return None

    out_dir = Path(out_dir) if out_dir is not None else run_dir / "plots"
    paths = plot_learning_curves(
        curves,
        out_dir=out_dir,
        x_label="Environment steps" if x == "env_step" else "Episode",
        y_label="Discounted return" if y == "discounted_return" else "Return",
        title=run_dir.name,
        formats=("png",),
    )
    return paths.get("png")
