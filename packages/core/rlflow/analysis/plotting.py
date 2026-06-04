from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_learning_curves(
    curves: pd.DataFrame,
    *,
    out_dir: str | Path,
    x_label: str = "Environment steps",
    y_label: str = "Discounted return",
    title: str | None = None,
    legend_title: str | None = None,
    width: float = 3.25,
    height: float = 2.35,
    dpi: int = 300,
    top_k: int | None = None,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if curves.empty:
        raise ValueError("No curve data to plot")

    plt = _pyplot()
    group_column = "group_id" if "group_id" in curves.columns else "group_key"
    group_keys = list(curves[group_column].drop_duplicates())
    if top_k is not None:
        group_keys = group_keys[:top_k]

    fig, ax = plt.subplots(figsize=(width, height))

    for group_key in group_keys:
        group = curves[curves[group_column] == group_key].sort_values("x")
        label = _curve_label(group, group_key)

        x = group["x"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        ci_low = group["ci_low"].to_numpy(dtype=float)
        ci_high = group["ci_high"].to_numpy(dtype=float)

        line = ax.plot(x, mean, linewidth=1.5, label=label)[0]
        ax.fill_between(x, ci_low, ci_high, alpha=0.18, color=line.get_color(), linewidth=0)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.legend(title=legend_title, frameon=False)
    fig.tight_layout()

    paths: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"learning_curve.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths[fmt] = path

    plt.close(fig)
    return paths


def _curve_label(group: pd.DataFrame, group_key: object) -> str:
    if "label" in group.columns:
        label = group["label"].iloc[0]
        if isinstance(label, str) and label:
            return label
    if "parameters" in group.columns:
        return _short_label(group["parameters"].iloc[0])
    return str(group_key)


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
        matplotlib.rcParams.update(
            {
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise RuntimeError(
                "matplotlib is required for plotting. Install analysis dependencies "
                "with `uv sync --extra analysis` or run with `uv run --extra analysis ...`."
            ) from exc
        raise
    return plt
