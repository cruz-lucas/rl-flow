from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import f_oneway

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_factorial_design.py"
SPEC = importlib.util.spec_from_file_location("analyze_factorial_design", SCRIPT_PATH)
assert SPEC is not None
factorial = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(factorial)


def coded_data(cells: dict[tuple[int, ...], list[float]]) -> pd.DataFrame:
    rows = []
    for cell, responses in cells.items():
        for response in responses:
            row = {factorial.RESPONSE_COLUMN: response}
            for index, value in enumerate(cell):
                row[factorial.coded_factor_column(f"factor_{index}")] = value
            rows.append(row)
    return pd.DataFrame(rows)


def test_welch_anova_matches_scipy_for_unequal_cells() -> None:
    cells = {
        (-1, -1): [1.0, 2.0, 4.0],
        (-1, 1): [2.0, 5.0, 9.0, 11.0],
        (1, -1): [6.0, 7.0, 12.0, 18.0, 20.0],
        (1, 1): [4.0, 9.0, 15.0, 22.0, 28.0, 35.0],
    }
    result = factorial.fit_welch_anova(
        coded_data(cells),
        factor_columns=["factor_0", "factor_1"],
        response_column=factorial.RESPONSE_COLUMN,
    )
    expected = f_oneway(*cells.values(), equal_var=False)

    assert result["method"] == "welch_one_way"
    assert result["grouping"] == "factorial_design_cells"
    assert result["factors"] == ["factor_0", "factor_1"]
    assert result["group_count"] == 4
    assert result["observation_count"] == 18
    assert result["df_numerator"] == 3.0
    assert result["df_denominator"] > 0
    assert result["f_statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        ({(-1,): [1.0], (1,): [2.0, 3.0]}, "at least two responses"),
        ({(-1,): [1.0, 1.0], (1,): [2.0, 3.0]}, "positive within-cell variance"),
        ({(-1,): [1.0, 2.0]}, "at least two observed design cells"),
    ],
)
def test_welch_anova_rejects_invalid_cells(
    cells: dict[tuple[int, ...], list[float]],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        factorial.fit_welch_anova(
            coded_data(cells),
            factor_columns=["factor_0"],
            response_column=factorial.RESPONSE_COLUMN,
        )


def test_welch_flag_controls_report_and_json_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = pd.DataFrame(
        {
            "factor_a": ["low"] * 3 + ["high"] * 3,
            factorial.RESPONSE_COLUMN: [1.0, 2.0, 4.0, 10.0, 12.0, 17.0],
            "source_label": ["experiment1"] * 3 + ["experiment2"] * 3,
            "sweep_id": ["sweep1"] * 3 + ["sweep2"] * 3,
            "seed_value": [1, 2, 3, 1, 2, 3],
        }
    )
    monkeypatch.setattr(
        factorial,
        "load_trial_responses",
        lambda *_args, **_kwargs: responses.copy(),
    )

    default_out = tmp_path / "default"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "dummy-sweep",
            "--factor",
            "factor_a",
            "--out",
            str(default_out),
            "--no-plots",
        ],
    )
    factorial.main()

    default_config = json.loads((default_out / "analysis_config.json").read_text())
    assert default_config["welch_anova"] is False
    assert not (default_out / "welch_anova.json").exists()
    assert "Welch ANOVA (Design Cells)" not in (default_out / "factorial_report.txt").read_text()

    welch_out = tmp_path / "welch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "dummy-sweep",
            "--factor",
            "factor_a",
            "--out",
            str(welch_out),
            "--no-plots",
            "--welch-anova",
        ],
    )
    factorial.main()

    welch_config = json.loads((welch_out / "analysis_config.json").read_text())
    result = json.loads((welch_out / "welch_anova.json").read_text())
    report = (welch_out / "factorial_report.txt").read_text()
    output = capsys.readouterr().out

    assert welch_config["welch_anova"] is True
    assert result["factors"] == ["factor_a"]
    assert result["group_count"] == 2
    assert result["observation_count"] == 6
    assert math.isfinite(result["f_statistic"])
    assert math.isfinite(result["df_denominator"])
    assert math.isfinite(result["p_value"])
    assert "Welch ANOVA (Design Cells)" in report
    assert f"welch_anova_json: {welch_out / 'welch_anova.json'}" in output
