from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_learning_curves(
    curves: pd.DataFrame,
    *,
    out_dir: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    top_k: int | None = None,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if curves.empty:
        raise ValueError("No curve data to plot")

    plt = _pyplot()
    group_keys = list(curves["group_key"].drop_duplicates())
    if top_k is not None:
        group_keys = group_keys[:top_k]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for group_key in group_keys:
        group = curves[curves["group_key"] == group_key].sort_values("x")
        label = _short_label(group["parameters"].iloc[0])

        x = group["x"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        ci_low = group["ci_low"].to_numpy(dtype=float)
        ci_high = group["ci_high"].to_numpy(dtype=float)

        line = ax.plot(x, mean, linewidth=2.0, label=label)[0]
        ax.fill_between(x, ci_low, ci_high, alpha=0.18, color=line.get_color(), linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    paths: dict[str, str] = {}
    for fmt in formats:
        path = out_dir / f"learning_curve.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths[fmt] = str(path)

    plt.close(fig)
    return paths


def _short_label(parameters: dict) -> str:
    if not parameters:
        return "{}"

    items: list[str] = []
    for key, value in parameters.items():
        short_key = key.split(".")[-1]
        items.append(f"{short_key}={value}")
    return ", ".join(items)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise RuntimeError(
                "matplotlib is required for plotting. Install analysis dependencies "
                "with `uv sync --extra analysis` or run with `uv run --extra analysis ...`."
            ) from exc
        raise
    return plt
