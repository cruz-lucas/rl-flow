from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from rlflow.analysis.loading import non_seed_parameters


def prepare_curve_dataframe(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    if x not in df.columns:
        if x == "env_step":
            raise ValueError(
                "Requested x='env_step', but histories do not contain env_step. "
                "Update runner logging first or use x='episode'."
            )
        raise ValueError(f"Missing x column: {x}")

    source_y = y
    if y not in df.columns:
        fallback = "return" if y == "discounted_return" else None
        if fallback and fallback in df.columns:
            source_y = fallback
        else:
            raise ValueError(f"Missing y column: {y}")

    required = ["trial_id", "parameters", x, source_y]
    extra = [c for c in ["seed_value", "group_id", "run_dir"] if c in df.columns]
    out = df[required + extra].copy()
    if source_y != y:
        out[y] = out[source_y]
        out = out[["trial_id", "parameters", x, y] + extra]
    out = out.dropna(subset=[x, y])
    out = out.sort_values(["trial_id", x])
    return out


def interpolate_trial_curve(
    trial_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    grid: np.ndarray,
) -> pd.DataFrame:
    trial_df = trial_df.sort_values(x)

    xs = trial_df[x].to_numpy(dtype=float)
    ys = trial_df[y].to_numpy(dtype=float)

    if len(xs) == 0:
        return pd.DataFrame()

    unique = pd.DataFrame({"x": xs, "y": ys}).groupby("x", as_index=False).last()
    xs = unique["x"].to_numpy(dtype=float)
    ys = unique["y"].to_numpy(dtype=float)

    if len(xs) == 1:
        valid_grid = grid[np.isclose(grid, xs[0])]
        values = np.full_like(valid_grid, ys[0], dtype=float)
    else:
        valid_grid = grid[(grid >= xs.min()) & (grid <= xs.max())]
        values = np.interp(valid_grid, xs, ys)

    if len(valid_grid) == 0:
        return pd.DataFrame()

    return pd.DataFrame({"x": valid_grid, "value": values})


def build_interpolated_curves(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    points: int,
) -> pd.DataFrame:
    if points < 2:
        raise ValueError("points must be at least 2")

    df = prepare_curve_dataframe(df, x=x, y=y)
    if df.empty:
        return pd.DataFrame()

    max_x = float(df[x].max())
    grid = np.linspace(0.0, max_x, points)

    rows: list[dict[str, Any]] = []
    for trial_id, trial_df in df.groupby("trial_id"):
        parameters = trial_df["parameters"].iloc[0]
        group_parameters = non_seed_parameters(parameters)
        group_key = json.dumps(group_parameters, sort_keys=True, default=str)
        interp = interpolate_trial_curve(trial_df, x=x, y=y, grid=grid)
        if interp.empty:
            continue
        seed_value = trial_df["seed_value"].iloc[0] if "seed_value" in trial_df.columns else None
        for _, row in interp.iterrows():
            rows.append(
                {
                    "trial_id": trial_id,
                    "group_key": group_key,
                    "parameters": group_parameters,
                    "seed_value": seed_value,
                    "x": float(row["x"]),
                    "value": float(row["value"]),
                }
            )

    return pd.DataFrame(rows)


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

    means: list[float] = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means.sort()

    low_index = min(len(means) - 1, max(0, int(0.025 * len(means))))
    high_index = min(len(means) - 1, max(0, int(0.975 * len(means))))
    return mean, float(means[low_index]), float(means[high_index])


def aggregate_interpolated_curves(
    interpolated: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if interpolated.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    grouped = interpolated.groupby(["group_key", "x"], sort=True)
    for (group_key, x_value), point_df in grouped:
        values = point_df["value"].to_numpy(dtype=float)
        mean, ci_low, ci_high = bootstrap_mean_ci(
            values,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        rows.append(
            {
                "group_key": group_key,
                "x": float(x_value),
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "seed_count": int(len(values)),
                "parameters": point_df["parameters"].iloc[0],
            }
        )

    return pd.DataFrame(rows)
