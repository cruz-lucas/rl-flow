import builtins
from pathlib import Path

import pandas as pd
import pytest

from rlflow.analysis.plotting import plot_learning_curves


def _curves() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_key": "{}",
                "x": 0.0,
                "mean": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
                "seed_count": 1,
                "parameters": {},
            },
            {
                "group_key": "{}",
                "x": 1.0,
                "mean": 2.0,
                "ci_low": 1.5,
                "ci_high": 2.5,
                "seed_count": 1,
                "parameters": {},
            },
        ]
    )


def test_plot_learning_curves_writes_requested_formats(tmp_path: Path) -> None:
    paths = plot_learning_curves(
        _curves(),
        out_dir=tmp_path,
        title="Learning Curve",
        x_label="Env Step",
        y_label="Return",
        formats=("png",),
    )
    assert set(paths) == {"png"}
    assert Path(paths["png"]).exists()


def test_plot_learning_curves_reports_missing_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="analysis dependencies"):
        plot_learning_curves(
            _curves(),
            out_dir=tmp_path,
            title="Learning Curve",
            x_label="Env Step",
            y_label="Return",
            formats=("png",),
        )
