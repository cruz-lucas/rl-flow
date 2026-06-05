from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlflow.analysis.loading import load_sweep_manifest, non_seed_parameters

_TRAIN_HISTORY_METRIC_RE = re.compile(
    r"^mean_train_(return|discounted_return)_last_(\d+)$"
)


def load_trial_metrics(manifest_path: str | Path) -> pd.DataFrame:
    compilation = load_sweep_manifest(manifest_path)
    rows: list[dict[str, Any]] = []

    for trial in compilation.trials:
        metrics_path, metrics = _load_metrics_for_trial(
            trial.metrics_path,
            trial.run_dir,
        )

        rows.append(
            {
                "trial_id": trial.trial_id,
                "group_id": trial.group_id,
                "seed_value": trial.seed_value,
                "run_dir": trial.run_dir,
                "metrics_path": trial.metrics_path,
                "parameters": trial.parameters,
                "group_parameters": non_seed_parameters(trial.parameters),
                "metrics": metrics,
            }
        )

    return pd.DataFrame(rows)


def _load_metrics_for_trial(metrics_path: str, run_dir: str) -> tuple[Path, dict[str, Any]]:
    candidates = [
        Path(metrics_path),
        Path(run_dir) / "summaries" / "metrics.json",
        Path(run_dir) / "metrics.json",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return path, data
    return Path(metrics_path), {}


def summarize_groups(
    manifest_path: str | Path,
    *,
    metric: str,
    goal: str = "maximize",
    metric_last_n: int | None = None,
) -> pd.DataFrame:
    if goal not in {"maximize", "minimize"}:
        raise ValueError("goal must be 'maximize' or 'minimize'")

    trials = load_trial_metrics(manifest_path)
    rows: list[dict[str, Any]] = []

    for _, row in trials.iterrows():
        value = _trial_metric_value(row, metric, metric_last_n)
        if not isinstance(value, (int, float)):
            continue

        group_key = json.dumps(row["group_parameters"], sort_keys=True, default=str)
        rows.append(
            {
                "trial_id": row["trial_id"],
                "run_dir": row["run_dir"],
                "group_id": row["group_id"],
                "group_key": group_key,
                "parameters": row["group_parameters"],
                "metric": float(value),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary_rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby("group_key"):
        values = group["metric"].to_numpy(dtype=float)
        group_id = _first_present(group["group_id"]) or f"group-{len(summary_rows):04d}"
        summary_rows.append(
            {
                "group_id": group_id,
                "group_key": group_key,
                "parameters": group["parameters"].iloc[0],
                "metric_mean": float(np.mean(values)),
                "metric_min": float(np.min(values)),
                "metric_max": float(np.max(values)),
                "metric_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "seed_count": int(len(values)),
                "trial_ids": list(group["trial_id"]),
                "run_dirs": list(group["run_dir"]),
            }
        )

    out = pd.DataFrame(summary_rows)
    ascending = goal == "minimize"
    out = out.sort_values("metric_mean", ascending=ascending).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def filter_top_k_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    if "group_id" in curves.columns and "group_id" in summary.columns:
        keep = set(summary.head(top_k)["group_id"])
        return curves[curves["group_id"].isin(keep)].copy()
    keep = set(summary.head(top_k)["group_key"])
    return curves[curves["group_key"].isin(keep)].copy()


def export_summary_tables(
    summary: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        raise ValueError("No summary data to export")

    export_df = summary.copy()
    for col in ["parameters", "trial_ids", "run_dirs"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: json.dumps(x, sort_keys=True, default=str)
            )

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    export_df.to_csv(csv_path, index=False)
    export_df.to_json(json_path, orient="records", indent=2)

    try:
        md_text = export_df.to_markdown(index=False)
    except Exception:
        md_text = _manual_markdown(export_df)

    md_path.write_text(md_text, encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def _manual_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _trial_metric_value(
    row: pd.Series,
    metric_name: str,
    metric_last_n: int | None,
) -> float | int | None:
    metrics = row["metrics"]
    if metric_name in metrics:
        value = metrics[metric_name]
        return value if isinstance(value, (int, float)) else None
    if metric_name == "mean_train_return":
        return _mean_train_history_metric(Path(row["run_dir"]), "return", None)
    if metric_name == "mean_train_return_last_n":
        return _mean_train_history_metric(Path(row["run_dir"]), "return", metric_last_n or 10)
    if metric_name == "mean_train_discounted_return":
        return _mean_train_history_metric(Path(row["run_dir"]), "discounted_return", None)
    if metric_name == "mean_train_discounted_return_last_n":
        return _mean_train_history_metric(
            Path(row["run_dir"]),
            "discounted_return",
            metric_last_n or 10,
        )
    match = _TRAIN_HISTORY_METRIC_RE.match(metric_name)
    if match is not None:
        return _mean_train_history_metric(
            Path(row["run_dir"]),
            match.group(1),
            int(match.group(2)),
        )
    return None


def _mean_train_history_metric(run_dir: Path, value_key: str, count: int | None) -> float | None:
    history_path = run_dir / "logs" / "train_history.jsonl"
    if not history_path.exists():
        return None
    values: list[float] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    window = values[-count:] if count is not None else values
    return float(np.mean(window))


def _first_present(series: pd.Series) -> str | None:
    for value in series:
        if value is None:
            continue
        try:
            missing = pd.isna(value)
        except TypeError:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and missing:
            continue
        text = str(value)
        if text:
            return text
    return None
