from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlflow.analysis.aggregation import (
    aggregate_interpolated_curves,
    bootstrap_mean_ci,
    build_interpolated_curves,
    prepare_curve_dataframe,
)


def test_interpolation_produces_shared_grid_points() -> None:
    df = pd.DataFrame(
        [
            {"trial_id": "trial-0000", "parameters": {"lr": 0.01}, "seed_value": 0, "episode": 0, "discounted_return": 1.0},
            {"trial_id": "trial-0000", "parameters": {"lr": 0.01}, "seed_value": 0, "episode": 10, "discounted_return": 2.0},
            {"trial_id": "trial-0001", "parameters": {"lr": 0.01}, "seed_value": 1, "episode": 0, "discounted_return": 1.5},
            {"trial_id": "trial-0001", "parameters": {"lr": 0.01}, "seed_value": 1, "episode": 15, "discounted_return": 3.0},
        ]
    )
    interpolated = build_interpolated_curves(df, x="episode", y="discounted_return", points=5)
    assert not interpolated.empty
    assert set(interpolated["x"].unique()) == set(np.linspace(0.0, 15.0, 5))
    assert interpolated[interpolated["trial_id"] == "trial-0000"].shape[0] == 3
    assert interpolated[interpolated["trial_id"] == "trial-0001"].shape[0] == 5

    curves = aggregate_interpolated_curves(interpolated, bootstrap_samples=0, seed=0)
    latest = curves[curves["x"] == 15.0].iloc[0]
    assert latest["seed_count"] == 1


def test_bootstrap_mean_ci_returns_finite_interval() -> None:
    values = np.array([1.0, 2.0, 3.0])
    mean, low, high = bootstrap_mean_ci(values, bootstrap_samples=100, rng=np.random.default_rng(0))
    assert mean == 2.0
    assert low <= mean <= high


def test_bootstrap_mean_ci_handles_single_seed_and_zero_samples() -> None:
    mean, low, high = bootstrap_mean_ci(
        np.array([4.0]),
        bootstrap_samples=100,
        rng=np.random.default_rng(0),
    )
    assert (mean, low, high) == (4.0, 4.0, 4.0)

    mean, low, high = bootstrap_mean_ci(
        np.array([1.0, 3.0]),
        bootstrap_samples=0,
        rng=np.random.default_rng(0),
    )
    assert (mean, low, high) == (2.0, 2.0, 2.0)


def test_discounted_return_falls_back_to_return_column() -> None:
    df = pd.DataFrame(
        [
            {"trial_id": "trial-0000", "parameters": {}, "episode": 0, "return": 1.0},
            {"trial_id": "trial-0000", "parameters": {}, "episode": 2, "return": 3.0},
        ]
    )

    prepared = prepare_curve_dataframe(df, x="episode", y="discounted_return")
    assert "discounted_return" in prepared.columns
    assert list(prepared["discounted_return"]) == [1.0, 3.0]


def test_missing_env_step_raises_clear_error() -> None:
    df = pd.DataFrame(
        [{"trial_id": "trial-0000", "parameters": {}, "episode": 0, "return": 1.0}]
    )
    with pytest.raises(ValueError, match="histories do not contain env_step"):
        prepare_curve_dataframe(df, x="env_step", y="return")
