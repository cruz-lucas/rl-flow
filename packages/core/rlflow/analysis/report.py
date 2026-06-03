from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def format_sweep_report(
    summary: dict[str, Any],
    *,
    top_k: int | None = 20,
    include_trials: bool = False,
) -> str:
    groups = list(summary.get("groups") or [])
    trials = list(summary.get("trials") or [])
    completed = [
        trial
        for trial in trials
        if isinstance(trial.get("metric"), (int, float))
    ]
    missing_count = len(trials) - len(completed)
    metric_last_n = summary.get("metric_last_n")

    lines = [
        f"Sweep: {summary.get('sweep_id', '-')}",
        _metric_line(summary),
        f"Trials: {len(completed)}/{len(trials)} completed, {missing_count} missing metric",
        f"Groups: {len(groups)} completed",
    ]

    if not groups:
        lines.extend(["", "No completed groups found for this metric."])
        return "\n".join(lines)

    if metric_last_n is not None:
        lines.append(f"Metric window: last {metric_last_n} training episodes")

    lines.extend(["", "Ranked Groups"])
    shown_groups = groups if top_k is None else groups[:top_k]
    lines.append(
        _render_table(
            [
                "rank",
                "group",
                "mean",
                "std",
                "min",
                "max",
                "n",
                "parameters",
            ],
            [
                [
                    str(index),
                    str(group.get("group_id", "-")),
                    _format_number(group.get("metric_mean", group.get("metric"))),
                    _format_number(group.get("metric_std")),
                    _format_number(group.get("metric_min")),
                    _format_number(group.get("metric_max")),
                    str(group.get("metric_count", "-")),
                    _format_parameters(group.get("parameters") or {}),
                ]
                for index, group in enumerate(shown_groups, start=1)
            ],
            right_aligned={"rank", "mean", "std", "min", "max", "n"},
        )
    )

    if top_k is not None and len(groups) > top_k:
        lines.append(f"... {len(groups) - top_k} more groups hidden; pass --all to show them.")

    if include_trials:
        lines.extend(["", "Trials"])
        lines.append(
            _render_table(
                ["trial", "group", "seed", "metric", "parameters", "run_dir"],
                [
                    [
                        str(trial.get("trial_id", "-")),
                        str(trial.get("group_id", "-")),
                        str(trial.get("seed_value", "-")),
                        _format_number(trial.get("metric")),
                        _format_parameters(trial.get("parameters") or {}),
                        str(trial.get("run_dir", "-")),
                    ]
                    for trial in trials
                ],
                right_aligned={"metric"},
            )
        )

    return "\n".join(lines)


def export_sweep_report(
    summary: dict[str, Any],
    *,
    out_dir: str | Path,
    top_k: int | None = 20,
    include_trials: bool = False,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "sweep_report.txt"
    summary_path = out_dir / "sweep_summary.json"
    groups_path = out_dir / "sweep_groups.csv"

    report_path.write_text(
        format_sweep_report(summary, top_k=top_k, include_trials=include_trials),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _write_groups_csv(groups_path, summary.get("groups") or [])

    return {
        "report": str(report_path),
        "summary_json": str(summary_path),
        "groups_csv": str(groups_path),
    }


def _metric_line(summary: dict[str, Any]) -> str:
    metric = summary.get("metric", "-")
    goal = summary.get("goal", "-")
    return f"Metric: {metric} ({goal})"


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.6g}"


def _format_parameters(parameters: dict[str, Any]) -> str:
    if not parameters:
        return "{}"
    return ", ".join(
        f"{key}={json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)}"
        for key, value in sorted(parameters.items())
    )


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right_aligned: set[str] | None = None,
) -> str:
    right_aligned = right_aligned or set()
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        if rows
        else len(header)
        for index, header in enumerate(headers)
    ]

    def render_row(cells: list[str]) -> str:
        parts = []
        for index, cell in enumerate(cells):
            header = headers[index]
            if header in right_aligned:
                parts.append(cell.rjust(widths[index]))
            else:
                parts.append(cell.ljust(widths[index]))
        return "  ".join(parts).rstrip()

    divider = ["-" * width for width in widths]
    return "\n".join([render_row(headers), render_row(divider), *(render_row(row) for row in rows)])


def _write_groups_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "group_id",
                "metric_mean",
                "metric_std",
                "metric_min",
                "metric_max",
                "metric_count",
                "parameters",
                "trial_ids",
                "run_dirs",
            ],
        )
        writer.writeheader()
        for rank, group in enumerate(groups, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "group_id": group.get("group_id"),
                    "metric_mean": group.get("metric_mean", group.get("metric")),
                    "metric_std": group.get("metric_std"),
                    "metric_min": group.get("metric_min"),
                    "metric_max": group.get("metric_max"),
                    "metric_count": group.get("metric_count"),
                    "parameters": json.dumps(group.get("parameters") or {}, sort_keys=True, default=str),
                    "trial_ids": json.dumps(group.get("trial_ids") or [], sort_keys=True, default=str),
                    "run_dirs": json.dumps(group.get("run_dirs") or [], sort_keys=True, default=str),
                }
            )
