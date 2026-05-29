import numpy as np
import pytest

from rlflow_api.services.dataset_analysis import offline_rnd_analysis


def test_offline_count_reference_ignores_symbolic_wall_distractors() -> None:
    observations = np.stack(
        [
            _symbolic_navix_observation(1, 1, wall_colour=12),
            _symbolic_navix_observation(1, 1, wall_colour=42),
            _symbolic_navix_observation(1, 1, wall_colour=99),
        ]
    )
    actions = np.asarray([0, 0, 1], dtype=np.int32)

    analysis = offline_rnd_analysis(
        observations,
        actions,
        algorithm="classifier",
        granularity="state_action",
        epochs=0,
        batch_size=2,
        learning_rate=0.001,
        hidden_units=(8,),
        activation="relu",
        optimizer="adam",
        action_conditioning="input",
        update_period=1,
        output_dim=4,
        intrinsic_reward_scale=1.0,
        intrinsic_stats_decay=0.99,
        intrinsic_reward_epsilon=1e-8,
        intrinsic_reward_clip=None,
        intrinsic_reward_center=False,
        max_grad_norm=1.0,
        seed=0,
    )

    assert analysis.unique_items == 3
    assert analysis.count_state_action_bonus is not None
    assert analysis.count_state_action_bonus[1][1][0] == pytest.approx(1.0 / np.sqrt(2.0))
    assert analysis.count_state_action_bonus[1][1][1] == pytest.approx(1.0)


def test_offline_simhash_analysis_returns_state_action_bonus() -> None:
    observations = np.stack(
        [
            _symbolic_navix_observation(1, 1, wall_colour=5),
            _symbolic_navix_observation(1, 1, wall_colour=5),
            _symbolic_navix_observation(1, 2, wall_colour=5),
        ]
    )
    actions = np.asarray([0, 0, 1], dtype=np.int32)

    analysis = offline_rnd_analysis(
        observations,
        actions,
        algorithm="simhash",
        granularity="state_action",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        hidden_units=(8,),
        activation="relu",
        optimizer="adam",
        action_conditioning="input",
        update_period=1,
        output_dim=4,
        intrinsic_reward_scale=1.0,
        intrinsic_stats_decay=0.99,
        intrinsic_reward_epsilon=1e-8,
        intrinsic_reward_clip=None,
        intrinsic_reward_center=False,
        max_grad_norm=1.0,
        seed=0,
        simhash_mode="static",
        simhash_bits=8,
        simhash_table_size=32,
    )

    assert analysis.algorithm == "simhash"
    assert analysis.learned_state_action_bonus is not None
    assert analysis.learned_state_action_bonus[1][1][0] is not None
    assert analysis.count_state_action_bonus is not None


def _symbolic_navix_observation(row: int, col: int, *, size: int = 5, wall_colour: int) -> np.ndarray:
    raw = np.zeros((size, size, 3), dtype=np.float32)
    raw[..., 0] = 1
    raw[0, :, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[-1, :, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[:, 0, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[:, -1, :] = np.asarray([2, wall_colour, 0], dtype=np.float32)
    raw[row, col, :] = np.asarray([10, 0, 0], dtype=np.float32)
    return (raw / 255.0).reshape(-1).astype(np.float32)
