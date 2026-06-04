import pytest
import pandas as pd

from rlflow.analysis.curves import (
    apply_curve_labels,
    interpolate_and_aggregate,
    smooth_curve_columns,
)


def test_interpolate_and_aggregate_uses_env_step_grid() -> None:
    df = pd.DataFrame(
        [
            {
                "trial_id": "trial-0000",
                "group_id": "group-0000",
                "seed_value": 0,
                "parameters": {"agent": "dqn", "seed": 0},
                "episode": 0,
                "env_step": 10,
                "discounted_return": 1.0,
            },
            {
                "trial_id": "trial-0000",
                "group_id": "group-0000",
                "seed_value": 0,
                "parameters": {"agent": "dqn", "seed": 0},
                "episode": 1,
                "env_step": 20,
                "discounted_return": 2.0,
            },
            {
                "trial_id": "trial-0001",
                "group_id": "group-0000",
                "seed_value": 1,
                "parameters": {"agent": "dqn", "seed": 1},
                "episode": 0,
                "env_step": 5,
                "discounted_return": 0.0,
            },
            {
                "trial_id": "trial-0001",
                "group_id": "group-0000",
                "seed_value": 1,
                "parameters": {"agent": "dqn", "seed": 1},
                "episode": 1,
                "env_step": 20,
                "discounted_return": 4.0,
            },
        ]
    )

    curves = interpolate_and_aggregate(
        df,
        x="env_step",
        y="discounted_return",
        points=3,
        bootstrap_samples=0,
    )

    at_ten = curves[curves["x"] == 10.0].iloc[0]
    assert at_ten["group_id"] == "group-0000"
    assert at_ten["seed_count"] == 2
    assert at_ten["mean"] == pytest.approx((1.0 + (5.0 / 15.0 * 4.0)) / 2.0)


def test_apply_curve_labels_accepts_parameter_json_keys() -> None:
    curves = pd.DataFrame(
        [
            {
                "group_id": "group-0000",
                "group_key": '{"agent": "dqn"}',
                "x": 1.0,
                "mean": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
                "seed_count": 1,
                "parameters": {"agent": "dqn"},
                "label": "agent=dqn",
            }
        ]
    )

    labeled = apply_curve_labels(curves, {'{"agent": "dqn"}': "DQN"})

    assert labeled.iloc[0]["label"] == "DQN"


def test_smooth_curve_columns_applies_trailing_window_after_aggregation() -> None:
    curves = pd.DataFrame(
        [
            {"group_id": "group-0000", "x": 0.0, "mean": 1.0, "ci_low": 0.0, "ci_high": 2.0},
            {"group_id": "group-0000", "x": 1.0, "mean": 3.0, "ci_low": 2.0, "ci_high": 4.0},
            {"group_id": "group-0000", "x": 2.0, "mean": 5.0, "ci_low": 4.0, "ci_high": 6.0},
        ]
    )

    smoothed = smooth_curve_columns(curves, window=2)

    assert list(smoothed["mean"]) == [1.0, 2.0, 4.0]
    assert list(smoothed["ci_low"]) == [0.0, 1.0, 3.0]
