from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("matplotlib")

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_tabular_rmax_emptyroom_maps.py"
SPEC = importlib.util.spec_from_file_location("plot_tabular_rmax_emptyroom_maps", SCRIPT_PATH)
assert SPEC is not None
plot_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_module)


def test_state_to_row_col_maps_tabular_emptyroom_inner_grid() -> None:
    assert plot_module.state_to_row_col(0, size=16) == (1, 1)
    assert plot_module.state_to_row_col(13, size=16) == (1, 14)
    assert plot_module.state_to_row_col(14, size=16) == (2, 1)
    assert plot_module.state_to_row_col(195, size=16) == (14, 14)


def test_fourrooms_valid_mask_keeps_openings_and_masks_middle_walls() -> None:
    mask = plot_module.valid_position_mask("fourrooms", 19)

    assert mask[1, 1]
    assert mask[17, 17]
    for row, col in [(4, 9), (14, 9), (9, 4), (9, 14)]:
        assert mask[row, col]
    for row, col in [(1, 9), (9, 1), (9, 9), (17, 9), (9, 17)]:
        assert not mask[row, col]


def test_state_action_to_grid_masks_fourrooms_middle_walls() -> None:
    values = np.ones((289, 4), dtype=np.float32)
    grid = plot_module.state_action_to_grid(values, size=19, env_name="four_rooms")

    assert np.all(np.isfinite(grid[4, 9]))
    assert np.all(np.isfinite(grid[9, 4]))
    assert np.all(np.isnan(grid[9, 9]))
    assert np.all(np.isnan(grid[1, 9]))
    assert np.all(np.isnan(grid[9, 1]))


def test_plot_tabular_rmax_emptyroom_maps_writes_three_pngs(tmp_path: Path) -> None:
    q_table = np.zeros((196, 4), dtype=np.float32)
    action_counts = np.zeros((196, 4), dtype=np.float32)
    q_table[0] = [1.0, 2.0, 3.0, 4.0]
    action_counts[0] = [1.0, 0.0, 2.0, 3.0]
    np.save(tmp_path / "q_table.npy", q_table)
    np.save(tmp_path / "action_counts.npy", action_counts)

    out_dir = tmp_path / "plots"
    assert plot_module.main([str(tmp_path), "--out-dir", str(out_dir), "--episodes", "5"]) == 0

    for filename in plot_module.plot_filenames(5).values():
        path = out_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_tabular_rmax_fourrooms_maps_writes_episode_png(tmp_path: Path) -> None:
    q_table = np.zeros((289, 4), dtype=np.float32)
    action_counts = np.zeros((289, 4), dtype=np.float32)
    q_table[0] = [1.0, 2.0, 3.0, 4.0]
    action_counts[0] = [1.0, 0.0, 2.0, 3.0]
    np.save(tmp_path / "q_table.npy", q_table)
    np.save(tmp_path / "action_counts.npy", action_counts)

    out_dir = tmp_path / "plots"
    assert plot_module.main(
        [
            str(tmp_path),
            "--env-name",
            "fourrooms",
            "--episodes",
            "7",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0

    q_plot = out_dir / "state_action_q_values_episode_7.png"
    assert q_plot.exists()
    assert q_plot.stat().st_size > 0
