from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
except ImportError as exc:  # pragma: no cover - exercised only when optional deps are missing.
    raise SystemExit(
        "matplotlib is required for plotting. Install the analysis/dev dependencies first."
    ) from exc


ACTION_LABELS = ("Up", "Down", "Left", "Right")
PLOT_FILENAMES = {
    "visitation": "state_action_visitation.png",
    "known": "state_action_known_unknown.png",
}
ENV_NAME_ALIASES = {
    "empty_room": "empty_room",
    "empty-room": "empty_room",
    "emptyroom": "empty_room",
    "four_rooms": "four_rooms",
    "four-rooms": "four_rooms",
    "fourrooms": "four_rooms",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot tabular R-Max Navix state-action maps from a run directory."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing q_table.npy and action_counts.npy")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/plots/tabular_rmax_<env>.",
    )
    parser.add_argument(
        "--env-name",
        default="empty_room",
        choices=sorted(ENV_NAME_ALIASES),
        help="Navix grid environment to render.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episode count represented by the final q_table.npy.",
    )
    parser.add_argument("--size", type=int, default=None, help="Navix grid size. Defaults to 16 for EmptyRoom, 19 for FourRooms.")
    parser.add_argument("--known-threshold", type=float, default=1.0, help="Count threshold for known pairs")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    env_name = normalize_env_name(args.env_name)
    size = args.size or default_size(env_name)
    filenames = plot_filenames(args.episodes)
    out_dir = args.out_dir or run_dir / "plots" / f"tabular_rmax_{env_slug(env_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    q_table = np.load(run_dir / "q_table.npy")
    action_counts = np.load(run_dir / "action_counts.npy")
    validate_tables(q_table, action_counts, size=size)

    visitation_grid = state_action_to_grid(action_counts, size=size, env_name=env_name)
    known_grid = state_action_to_grid((action_counts >= args.known_threshold).astype(float), size=size, env_name=env_name)
    q_grid = state_action_to_grid(q_table, size=size, env_name=env_name)

    plot_state_action_map(
        visitation_grid,
        out_dir / filenames["visitation"],
        title="State-Action Visitation",
        value_label="visits",
        cmap=plt.get_cmap("YlGnBu"),
        norm=count_norm(visitation_grid),
        colorbar_ticks=None,
    )
    plot_state_action_map(
        known_grid,
        out_dir / filenames["known"],
        title="Known / Unknown State-Actions",
        value_label="known",
        cmap=mcolors.ListedColormap(["#f8fafc", "#166534"]),
        norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2),
        colorbar_ticks=[0, 1],
        colorbar_ticklabels=["Unknown", "Known"],
    )
    plot_state_action_map(
        q_grid,
        out_dir / filenames["q_values"],
        title=f"Q-Values After {args.episodes} Episodes",
        value_label="Q",
        cmap=plt.get_cmap("viridis"),
        norm=value_norm(q_grid),
        colorbar_ticks=None,
    )

    for filename in filenames.values():
        print(out_dir / filename)
    return 0


def normalize_env_name(env_name: str) -> str:
    return ENV_NAME_ALIASES[env_name]


def default_size(env_name: str) -> int:
    if env_name == "four_rooms":
        return 19
    return 16


def env_slug(env_name: str) -> str:
    if env_name == "four_rooms":
        return "fourrooms"
    return "emptyroom"


def plot_filenames(episodes: int) -> dict[str, str]:
    return {
        **PLOT_FILENAMES,
        "q_values": f"state_action_q_values_episode_{episodes}.png",
    }


def validate_tables(q_table: np.ndarray, action_counts: np.ndarray, *, size: int) -> None:
    expected_states = (size - 2) * (size - 2)
    expected_shape = (expected_states, len(ACTION_LABELS))
    if q_table.shape != expected_shape:
        raise ValueError(f"Expected q_table shape {expected_shape}, got {q_table.shape}")
    if action_counts.shape != expected_shape:
        raise ValueError(f"Expected action_counts shape {expected_shape}, got {action_counts.shape}")


def state_action_to_grid(values: np.ndarray, *, size: int = 16, env_name: str = "empty_room") -> np.ndarray:
    env_name = normalize_env_name(env_name)
    values = np.asarray(values, dtype=float)
    expected_states = (size - 2) * (size - 2)
    if values.shape != (expected_states, len(ACTION_LABELS)):
        raise ValueError(
            f"Expected values shape {(expected_states, len(ACTION_LABELS))}, got {values.shape}"
        )

    grid = np.full((size, size, len(ACTION_LABELS)), np.nan, dtype=float)
    for state in range(expected_states):
        row, col = state_to_row_col(state, size=size)
        grid[row, col, :] = values[state, :]
    grid[~valid_position_mask(env_name, size), :] = np.nan
    return grid


def state_to_row_col(state: int, *, size: int = 16) -> tuple[int, int]:
    inner_width = size - 2
    return state // inner_width + 1, state % inner_width + 1


def valid_position_mask(env_name: str, size: int) -> np.ndarray:
    env_name = normalize_env_name(env_name)
    mask = np.zeros((size, size), dtype=bool)
    mask[1 : size - 1, 1 : size - 1] = True
    if env_name == "four_rooms":
        if size != 19:
            raise ValueError("FourRooms plotting expects the canonical 19x19 grid")
        center = size // 2
        openings = {4, 14}
        mask[center, 1 : size - 1] = False
        mask[1 : size - 1, center] = False
        for opening in openings:
            mask[center, opening] = True
            mask[opening, center] = True
    return mask


def count_norm(values: np.ndarray) -> mcolors.Normalize:
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    return mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))


def value_norm(values: np.ndarray) -> mcolors.Normalize:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.1, 1.0)
        return mcolors.Normalize(vmin=vmin - padding, vmax=vmax + padding)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_state_action_map(
    grid: np.ndarray,
    path: Path,
    *,
    title: str,
    value_label: str,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    colorbar_ticks: list[int] | None,
    colorbar_ticklabels: list[str] | None = None,
) -> None:
    size = grid.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for row in range(size):
        for col in range(size):
            values = grid[row, col]
            if not np.any(np.isfinite(values)):
                ax.add_patch(
                    Rectangle(
                        (col, row),
                        1,
                        1,
                        facecolor="#ffffff",
                        edgecolor="#e5e7eb",
                        linewidth=0.4,
                    )
                )
                continue
            for action, points in enumerate(action_polygons(row, col)):
                value = values[action]
                facecolor = "#ffffff" if not np.isfinite(value) else cmap(norm(value))
                patch = Polygon(
                    points,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor="#4b5563",
                    linewidth=0.25,
                )
                ax.add_patch(patch)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(value_label)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        colorbar.set_ticklabels(colorbar_ticklabels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def action_polygons(row: int, col: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    center = (col + 0.5, row + 0.5)
    return (
        ((col, row), (col + 1, row), center),
        ((col, row + 1), (col + 1, row + 1), center),
        ((col, row), (col, row + 1), center),
        ((col + 1, row), (col + 1, row + 1), center),
    )


if __name__ == "__main__":
    raise SystemExit(main())
