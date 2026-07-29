"""Paper-ready boxplot for the Strehl & Littman factorial + comparison study.

Reads per-seed ``cumulative_train_reward`` from one or more compiled sweep manifests
(each seed-only sweep is one experimental condition) -- algorithm on the x-axis, cumulative
reward on the y-axis. Matches the house matplotlib style (fonttype-42, dpi 300), exports
pdf/svg/png, and writes a ``<stem>_summary.csv`` with the mean and a bootstrap 95% CI of
the mean per condition.

By default it draws a traditional boxplot (box, whiskers, outliers); pass ``--points`` to
overlay every seed as a jittered point instead.

Any set of conditions can be plotted -- pass whichever sweep dirs you want. Set a custom
x-axis label for a condition with ``LABEL=PATH``:

    # the full study (16 factorial cells + 4 comparison algorithms)
    uv run python scripts/plot_factorial_boxplot.py --env riverswim \\
        --factorial runs/sweeps/sl/riverswim/factorial/experiment* \\
        --compare   runs/sweeps/sl/riverswim/compare/* \\
        --design    configs/sweeps/sl/riverswim/factorial_design.csv \\
        --out       runs/analysis/sl_boxplot/riverswim

    # a few conditions, with custom labels and individual points
    uv run python scripts/plot_factorial_boxplot.py --env riverswim --points \\
        --sweeps "R-Max (m=16)=runs/sweeps/sl/riverswim/compare/rmax" \\
                 "MBIE-EB=runs/sweeps/sl/riverswim/compare/mbie_eb" \\
                 "Best cell (O+C+R)=runs/sweeps/sl/riverswim/factorial/experiment15" \\
        --out    runs/analysis/sl_boxplot/riverswim_subset
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from rlflow.analysis.loading import load_sweep_manifest
from rlflow.analysis.plotting import _pyplot
from rlflow.analysis.summary import load_trial_metrics

FACTOR_LETTERS = {
    "epsilon_greedy": "E",
    "optimistic_init": "O",
    "count_bonus": "C",
    "replay": "R",
}
COMPARISON_LABELS = {
    "rmax": "R-Max",
    "mbie_eb": "MBIE-EB",
    "replay_rmax": "Replay R-Max",
    "replay_mbie_eb": "Replay MBIE-EB",
}
COMPARISON_ORDER = ["rmax", "mbie_eb", "replay_rmax", "replay_mbie_eb"]

FACTORIAL_COLOR = "#4c78a8"
COMPARISON_COLOR = "#f58518"


def _manifest_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        candidate = path / "sweep_manifest.yaml"
        if candidate.is_file():
            return candidate
        matches = sorted(path.glob("**/sweep_manifest.yaml"))
        if matches:
            return matches[0]
    return path


def _values(manifest: Path, metric: str) -> np.ndarray:
    frame = load_trial_metrics(manifest)
    values = [
        row.get(metric)
        for row in frame["metrics"]
        if isinstance(row, dict) and row.get(metric) is not None
    ]
    return np.asarray([float(v) for v in values], dtype=float)


def _sweep_name(manifest: Path) -> str:
    try:
        return load_sweep_manifest(manifest).name or str(manifest)
    except Exception:  # noqa: BLE001 - keep plotting even if a manifest is malformed
        return str(manifest)


def _experiment_number(name: str) -> int | None:
    match = re.search(r"experiment(\d+)", name)
    return int(match.group(1)) if match else None


def _compare_key(name: str) -> str | None:
    for key in sorted(COMPARISON_LABELS, key=len, reverse=True):
        if f"compare__{key}" in name or f"__{key}_" in name or name.endswith(f"__{key}"):
            return key
    return None


def _load_design(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    labels: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            number = int(row["experiment_number"])
            letters = "".join(
                letter
                for factor, letter in FACTOR_LETTERS.items()
                if str(row.get(factor, "0")) == "1"
            )
            labels[number] = letters or "base"
    return labels


class Condition:
    __slots__ = ("label", "values", "color", "kind", "sort_key")

    def __init__(self, label, values, color, kind, sort_key):
        self.label = label
        self.values = values
        self.color = color
        self.kind = kind
        self.sort_key = sort_key


def _split_label(raw: str) -> tuple[str | None, str]:
    """Split a ``LABEL=PATH`` entry into ``(label, path)``; a plain path -> ``(None, path)``.

    This is how you set a custom x-axis name for a condition, e.g.
    ``--sweeps "R-Max (m=16)=runs/sweeps/sl/riverswim/compare/rmax"``.
    """
    # rpartition on the LAST '=' so labels may themselves contain '=' (e.g. "R-Max (m=16)").
    if "=" in raw:
        label, _, path = raw.rpartition("=")
        if label and ("/" in path or Path(path).exists()):
            return label, path
    return None, raw


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (robust to the skewed reward distributions)."""
    n = values.size
    if n < 2:
        mean = float(values.mean()) if n else float("nan")
        return mean, mean
    rng = np.random.default_rng(seed)
    resamples = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(resamples, 2.5)), float(np.percentile(resamples, 97.5))


def _condition(raw: str, metric: str, design: dict[int, str], input_index: int) -> Condition | None:
    override, raw_path = _split_label(raw)
    manifest = _manifest_path(raw_path)
    name = _sweep_name(manifest)
    values = _values(manifest, metric)
    if not values.size:
        print(f"  skip (no {metric} values): {raw_path}")
        return None

    number = _experiment_number(name)
    if number is not None:
        tag = design.get(number, "")
        label = override or (f"{number}·{tag}" if tag else f"cell {number}")
        return Condition(label, values, FACTORIAL_COLOR, "factorial", (0, number))

    key = _compare_key(name)
    if key is not None:
        order = COMPARISON_ORDER.index(key) if key in COMPARISON_ORDER else 99
        return Condition(
            override or COMPARISON_LABELS[key], values, COMPARISON_COLOR, "compare", (1, order)
        )

    return Condition(override or name, values, FACTORIAL_COLOR, "other", (2, input_index))


def _collect(args) -> list[Condition]:
    design = _load_design(Path(args.design) if args.design else None)
    conditions: list[Condition] = []
    for i, raw in enumerate([*args.factorial, *args.compare, *args.sweeps]):
        cond = _condition(raw, args.metric, design, i)
        if cond is not None:
            conditions.append(cond)
    conditions.sort(key=lambda c: c.sort_key)
    return conditions


def _plot(conditions: list[Condition], args, stem: str) -> dict[str, Path]:
    plt = _pyplot()
    n = len(conditions)
    width = args.width if args.width else max(3.25, 0.62 * n + 1.2)
    fig, ax = plt.subplots(figsize=(width, args.height))
    positions = list(range(1, n + 1))
    data = [c.values for c in conditions]

    # Default: a traditional boxplot (box, whiskers, outliers). With --points we instead
    # overlay every seed as a jittered point and drop the fliers (they'd be redundant).
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=not args.points,
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markerfacecolor": "0.35",
            "markeredgecolor": "none",
        },
        medianprops={"color": "black", "linewidth": 1.1, "zorder": 4},
        whiskerprops={"color": "0.4", "linewidth": 0.7},
        capprops={"color": "0.4", "linewidth": 0.7},
        boxprops={"zorder": 2},
        zorder=2,
    )
    for patch, cond in zip(box["boxes"], conditions, strict=True):
        patch.set_facecolor(cond.color)
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    if args.points:
        rng = np.random.default_rng(0)
        for pos, cond in zip(positions, conditions, strict=True):
            jitter = rng.uniform(-args.jitter, args.jitter, size=cond.values.size)
            ax.scatter(
                pos + jitter,
                cond.values,
                s=6,
                color=cond.color,
                alpha=0.5,
                linewidths=0,
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [c.label for c in conditions], rotation=args.rotation, ha="right", fontfamily="monospace"
    )
    ax.set_ylabel(f"Cumulative reward")
    if args.title:
        ax.set_title(args.title)
    elif args.env:
        ax.set_title(args.env)

    values_min = min(float(np.min(c.values)) for c in conditions)
    scale = args.scale
    if scale == "auto":
        scale = "log" if values_min > 0 else "symlog"
    if scale == "log" and values_min > 0:
        ax.set_yscale("log")
    elif scale in ("log", "symlog"):
        # Rewards are non-negative but some seeds hit exactly 0, so use symlog with a
        # small linear region near 0 and crop the (unused) negative decades.
        linthresh = max(1.0, args.linthresh)
        ax.set_yscale("symlog", linthresh=linthresh)
        if values_min >= 0:
            ax.set_ylim(bottom=-linthresh)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.5)
    ax.margins(x=0.02)
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = args.dpi
        fig.savefig(path, **save_kwargs)
        paths[fmt] = path
    plt.close(fig)
    return paths


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--factorial", nargs="*", default=[], help="Factorial sweep manifest dirs")
    parser.add_argument("--compare", nargs="*", default=[], help="Comparison sweep manifest dirs")
    parser.add_argument(
        "--sweeps",
        nargs="*",
        default=[],
        help="Any additional sweep manifest dirs (auto-labeled). Use this to plot a subset.",
    )
    parser.add_argument("--design", default=None, help="factorial_design.csv for cell labels")
    parser.add_argument("--env", default="", help="Environment name (title + filename)")
    parser.add_argument("--out", required=True, help="Output directory for the figure + CSV")
    parser.add_argument(
        "--csv", default=None, help="Summary CSV path (default: <out>/<stem>_summary.csv)"
    )
    parser.add_argument("--metric", default="cumulative_train_reward")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--width", type=float, default=0.0, help="Figure width (0 = auto from #conditions)"
    )
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rotation", type=float, default=40.0, help="x tick label rotation")
    parser.add_argument(
        "--points",
        action="store_true",
        help="Overlay every seed as a jittered point (and hide fliers). "
        "Omit for a traditional boxplot with box, whiskers and outliers.",
    )
    parser.add_argument("--jitter", type=float, default=0.16, help="point jitter half-width")
    parser.add_argument(
        "--linthresh", type=float, default=1.0, help="symlog linear-region threshold near 0"
    )
    parser.add_argument(
        "--scale",
        choices=("auto", "log", "symlog", "linear"),
        default="auto",
        help="y-axis scale (auto: log when all values are positive, else symlog)",
    )
    return parser.parse_args(argv)


def _write_summary_csv(conditions: list[Condition], path: Path) -> None:
    """Per-condition mean, bootstrap 95% CI of the mean, and distribution summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "condition",
        "kind",
        "n",
        "mean",
        "std",
        "sem",
        "ci95_low",
        "ci95_high",
        "median",
        "min",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for cond in conditions:
            values = cond.values
            n = int(values.size)
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci(values)
            writer.writerow(
                [
                    cond.label,
                    cond.kind,
                    n,
                    float(values.mean()),
                    std,
                    std / np.sqrt(n) if n else 0.0,
                    ci_low,
                    ci_high,
                    float(np.median(values)),
                    float(values.min()),
                    float(values.max()),
                ]
            )


def main(argv=None) -> int:
    args = _parse_args(argv)
    conditions = _collect(args)
    if not conditions:
        raise SystemExit("No conditions with data were found; check the manifest paths.")

    stem = f"boxplot_{args.env}" if args.env else "boxplot"
    print(f"{'condition':16s}  n   mean        95% CI")
    for cond in conditions:
        lo, hi = _bootstrap_ci(cond.values)
        print(
            f"{cond.label:16s}  {cond.values.size:>3d}  {cond.values.mean():>10.1f}  [{lo:.1f}, {hi:.1f}]"
        )

    paths = _plot(conditions, args, stem)
    csv_path = Path(args.csv) if args.csv else Path(args.out) / f"{stem}_summary.csv"
    _write_summary_csv(conditions, csv_path)
    for fmt, path in paths.items():
        print(f"wrote {fmt}: {path}")
    print(f"wrote csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
