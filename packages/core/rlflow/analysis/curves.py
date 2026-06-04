from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rlflow.analysis.loading import non_seed_parameters


def interpolate_and_aggregate(
    df: pd.DataFrame,
    *,
    x: str = "env_step",
    y: str = "discounted_return",
    points: int = 500,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    interpolated = interpolate_seed_curves(df, x=x, y=y, points=points)
    return aggregate_seed_curves(
        interpolated,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def interpolate_seed_curves(
    df: pd.DataFrame,
    *,
    x: str = "env_step",
    y: str = "discounted_return",
    points: int = 500,
) -> pd.DataFrame:
    if points < 2:
        raise ValueError("points must be at least 2")

    prepared = prepare_curve_dataframe(df, x=x, y=y)
    if prepared.empty:
        return pd.DataFrame(columns=_interpolated_columns())

    min_x = float(prepared[x].min())
    max_x = float(prepared[x].max())
    start_x = 0.0 if min_x >= 0.0 else min_x
    grid = np.linspace(start_x, max_x, points)

    rows: list[dict[str, Any]] = []
    seed_groups = prepared.groupby(["group_id", "seed_id"], sort=True)
    for (group_id, seed_id), seed_df in seed_groups:
        seed_df = seed_df.sort_values(x)
        unique = seed_df.groupby(x, as_index=False).last()
        xs = unique[x].to_numpy(dtype=float)
        ys = unique[y].to_numpy(dtype=float)
        if len(xs) == 0:
            continue
        if len(xs) == 1:
            valid_grid = grid[np.isclose(grid, xs[0])]
            values = np.full_like(valid_grid, ys[0], dtype=float)
        else:
            valid_grid = grid[(grid >= xs.min()) & (grid <= xs.max())]
            values = np.interp(valid_grid, xs, ys)
        if len(valid_grid) == 0:
            continue

        first = seed_df.iloc[0]
        for x_value, value in zip(valid_grid, values, strict=True):
            rows.append(
                {
                    "group_id": str(group_id),
                    "group_key": first["group_key"],
                    "seed_id": str(seed_id),
                    "seed_value": first["seed_value"],
                    "trial_id": first["trial_id"],
                    "x": float(x_value),
                    "value": float(value),
                    "parameters": first["group_parameters"],
                }
            )

    return pd.DataFrame(rows, columns=_interpolated_columns())


def aggregate_seed_curves(
    interpolated: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    if interpolated.empty:
        return pd.DataFrame(columns=_aggregate_columns())

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (group_id, x_value), point_df in interpolated.groupby(["group_id", "x"], sort=True):
        seed_values = point_df.groupby("seed_id")["value"].mean().to_numpy(dtype=float)
        mean, ci_low, ci_high = bootstrap_mean_ci(
            seed_values,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        parameters = point_df["parameters"].iloc[0]
        rows.append(
            {
                "group_id": str(group_id),
                "group_key": point_df["group_key"].iloc[0],
                "x": float(x_value),
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "seed_count": int(len(seed_values)),
                "parameters": parameters,
                "label": default_curve_label(parameters),
            }
        )

    return pd.DataFrame(rows, columns=_aggregate_columns())


def prepare_curve_dataframe(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if x not in df.columns:
        if x == "env_step":
            raise ValueError(
                "Requested x='env_step', but histories do not contain env_step. "
                "Update runner logging first or use x='episode'."
            )
        raise ValueError(f"Missing x column: {x}")

    source_y = y
    work = df.copy()
    if y not in work.columns:
        if y == "discounted_return" and "return" in work.columns:
            work[y] = work["return"]
        else:
            raise ValueError(f"Missing y column: {y}")
    elif y == "discounted_return" and "return" in work.columns:
        work[y] = work[y].fillna(work["return"])

    required = ["trial_id", "parameters", x, source_y]
    for column in required:
        if column not in work.columns:
            raise ValueError(f"Missing required history column: {column}")

    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[x, y])
    if work.empty:
        return pd.DataFrame()

    for column in ["group_id", "seed_value", "run_dir"]:
        if column not in work.columns:
            work[column] = None

    work["group_parameters"] = work["parameters"].apply(_group_parameters)
    work["group_key"] = work["group_parameters"].apply(_stable_json)
    group_ids = _group_id_map(work)
    work["group_id"] = work["group_key"].map(group_ids)
    work["seed_id"] = work.apply(_seed_id, axis=1)

    columns = [
        "trial_id",
        "group_id",
        "group_key",
        "seed_id",
        "seed_value",
        "run_dir",
        "parameters",
        "group_parameters",
        x,
        y,
    ]
    return work[columns].sort_values(["group_id", "seed_id", x])


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("No values to bootstrap")

    mean = float(np.mean(values))
    if len(values) == 1 or bootstrap_samples <= 0:
        return mean, mean, mean

    samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
    means = samples.mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return mean, float(ci_low), float(ci_high)


def apply_curve_labels(
    curves: pd.DataFrame,
    labels: Mapping[str, str] | None,
) -> pd.DataFrame:
    if curves.empty:
        return curves.copy()

    out = curves.copy()
    normalized = _normalized_labels(labels or {})
    out["label"] = [
        resolve_curve_label(row.group_id, row.parameters, normalized)
        for row in out.itertuples(index=False)
    ]
    return out


def smooth_curve_columns(curves: pd.DataFrame, *, window: int | None) -> pd.DataFrame:
    if curves.empty or window is None or window <= 1:
        return curves.copy()

    out = curves.copy()
    smoothed_groups: list[pd.DataFrame] = []
    for _, group in out.groupby("group_id", sort=False):
        group = group.sort_values("x").copy()
        for column in ["mean", "ci_low", "ci_high"]:
            group[column] = group[column].rolling(window=window, min_periods=1).mean()
        smoothed_groups.append(group)
    return pd.concat(smoothed_groups, ignore_index=True)


def default_curve_label(parameters: Mapping[str, Any]) -> str:
    if not parameters:
        return "{}"
    return ", ".join(
        f"{str(key).split('.')[-1]}={value}"
        for key, value in parameters.items()
    )


def resolve_curve_label(
    group_id: str,
    parameters: Mapping[str, Any],
    labels: Mapping[str, str],
) -> str:
    if group_id in labels:
        return labels[group_id]
    parameter_key = _stable_json(parameters)
    if parameter_key in labels:
        return labels[parameter_key]
    compact_key = json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":"))
    if compact_key in labels:
        return labels[compact_key]
    return default_curve_label(parameters)


def _group_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    return non_seed_parameters(parameters)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _group_id_map(df: pd.DataFrame) -> dict[str, str]:
    used: set[str] = set()
    mapping: dict[str, str] = {}

    for group_key, group in df.groupby("group_key", sort=True):
        existing = [
            str(value)
            for value in group["group_id"]
            if not _is_missing(value)
        ]
        if existing:
            mapping[group_key] = existing[0]
            used.add(existing[0])

    next_index = 0
    for group_key in sorted(df["group_key"].unique()):
        if group_key in mapping:
            continue
        while True:
            candidate = f"group-{next_index:04d}"
            next_index += 1
            if candidate not in used:
                mapping[group_key] = candidate
                used.add(candidate)
                break

    return mapping


def _seed_id(row: pd.Series) -> str:
    seed_value = row.get("seed_value")
    if not _is_missing(seed_value):
        return _stable_json(seed_value)
    return str(row["trial_id"])


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        missing = pd.isna(value)
    except TypeError:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _normalized_labels(labels: Mapping[str, str]) -> dict[str, str]:
    normalized = dict(labels)
    for key, value in labels.items():
        try:
            decoded = json.loads(key)
        except (TypeError, json.JSONDecodeError):
            continue
        normalized[_stable_json(decoded)] = value
        normalized[json.dumps(decoded, sort_keys=True, default=str, separators=(",", ":"))] = value
    return normalized


def _interpolated_columns() -> list[str]:
    return [
        "group_id",
        "group_key",
        "seed_id",
        "seed_value",
        "trial_id",
        "x",
        "value",
        "parameters",
    ]


def _aggregate_columns() -> list[str]:
    return [
        "group_id",
        "group_key",
        "x",
        "mean",
        "ci_low",
        "ci_high",
        "seed_count",
        "parameters",
        "label",
    ]
