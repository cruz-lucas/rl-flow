from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlflow.analysis.loading import load_sweep_manifest, non_seed_parameters


def load_trial_metrics(manifest_path: str | Path) -> pd.DataFrame:
    compilation = load_sweep_manifest(manifest_path)
    rows: list[dict[str, Any]] = []

    for trial in compilation.trials:
        metrics_path = Path(trial.metrics_path)
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "trial_id": trial.trial_id,
                "run_dir": trial.run_dir,
                "metrics_path": trial.metrics_path,
                "parameters": trial.parameters,
                "group_parameters": non_seed_parameters(trial.parameters),
                "metrics": metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_groups(
    manifest_path: str | Path,
    *,
    metric: str,
    goal: str = "maximize",
) -> pd.DataFrame:
    if goal not in {"maximize", "minimize"}:
        raise ValueError("goal must be 'maximize' or 'minimize'")

    trials = load_trial_metrics(manifest_path)
    rows: list[dict[str, Any]] = []

    for _, row in trials.iterrows():
        value = row["metrics"].get(metric)
        if not isinstance(value, (int, float)):
            continue

        group_key = json.dumps(row["group_parameters"], sort_keys=True, default=str)
        rows.append(
            {
                "trial_id": row["trial_id"],
                "run_dir": row["run_dir"],
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
        summary_rows.append(
            {
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
