from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from rlflow.analysis.loading import load_histories, load_sweep_manifest, non_seed_parameters


# Optional editable design table. Prefer passing --design path.csv once the table
# is finalized, but this is useful while iterating.
#
# Example:
# EMBEDDED_DESIGN = [
#     {"experiment": 1, "factor_a": -1, "factor_b": -1, "factor_c": -1},
#     {"experiment": 2, "factor_a": +1, "factor_b": -1, "factor_c": -1},
# ]
EMBEDDED_DESIGN: list[dict[str, Any]] = []

RESPONSE_COLUMN = "average_discounted_return"
JOIN_PRIORITY = (
    "experiment_number",
    "experiment",
    "design_id",
    "sweep_id",
    "source_label",
    "sweep_name",
    "workflow",
    "workflow_path",
)
NON_FACTOR_COLUMNS = {
    "comment",
    "description",
    "design_join_column",
    "design_id",
    "experiment",
    "experiment_id",
    "experiment_number",
    "group_id",
    "group_key",
    "history",
    "manifest_path",
    "name",
    "notes",
    "response",
    RESPONSE_COLUMN,
    "response_count",
    "response_source",
    "run_dir",
    "seed",
    "seed_count",
    "seed_value",
    "source_label",
    "sweep",
    "sweep_dir",
    "sweep_id",
    "sweep_name",
    "trial_id",
    "workflow",
    "workflow_path",
}
LOW_LEVELS = {"-1", "-", "0", "false", "f", "low", "lo", "l", "no", "n", "off"}
HIGH_LEVELS = {"+1", "1", "+", "true", "t", "high", "hi", "h", "yes", "y", "on"}


def main() -> None:
    args = _parse_args()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = load_trial_responses(
        args.sweeps,
        history=args.history,
        response=args.response,
        last_n=args.last_n,
        fallback_to_return=args.fallback_to_return,
    )
    if responses.empty:
        raise SystemExit("No trial responses found. Check sweep paths and history files.")

    design, design_source = load_design_table(args.design)
    data = merge_design(responses, design, join_column=args.join_column)
    factor_columns = resolve_factor_columns(
        data,
        requested=parse_factor_columns(args.factor),
        design_columns=list(design.columns) if design is not None else None,
    )

    warnings: list[str] = []
    if not factor_columns:
        warnings.append(
            "No two-level factor columns were found. Wrote response summaries only."
        )
        coded = data.copy()
        coding: dict[str, Any] = {}
        effects = pd.DataFrame()
        model_summary: dict[str, Any] = {}
    else:
        coded, coding = code_factor_columns(data, factor_columns)
        max_order = parse_max_order(args.max_order, len(factor_columns))
        fit = fit_factorial_model(
            coded,
            factor_columns=factor_columns,
            max_order=max_order,
            response_column=RESPONSE_COLUMN,
        )
        effects = fit["effects"]
        model_summary = fit["model_summary"]
        warnings.extend(fit["warnings"])

    cell_summary = summarize_cells(coded, factor_columns)

    response_csv = out_dir / "trial_responses.csv"
    cell_csv = out_dir / "cell_summary.csv"
    effects_csv = out_dir / "effects.csv"
    coding_json = out_dir / "factor_coding.json"
    report_txt = out_dir / "factorial_report.txt"
    config_json = out_dir / "analysis_config.json"

    write_dataframe_csv(coded.drop(columns=internal_columns(coded)), response_csv)
    write_dataframe_csv(cell_summary, cell_csv)
    write_dataframe_csv(effects, effects_csv)
    coding_json.write_text(json.dumps(coding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_json.write_text(
        json.dumps(
            {
                "sweeps": [str(path) for path in args.sweeps],
                "design_source": design_source,
                "history": args.history,
                "response": args.response,
                "last_n": args.last_n,
                "join_column": args.join_column,
                "factor_columns": factor_columns,
                "max_order": args.max_order,
                "fallback_to_return": args.fallback_to_return,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_txt.write_text(
        format_report(
            responses=coded,
            cell_summary=cell_summary,
            effects=effects,
            factor_columns=factor_columns,
            model_summary=model_summary,
            warnings=warnings,
        ),
        encoding="utf-8",
    )

    plot_paths: list[Path] = []
    if factor_columns and not args.no_plots:
        plot_paths = write_plots(effects, coded, factor_columns, out_dir=out_dir)

    print(f"trial_responses_csv: {response_csv}")
    print(f"cell_summary_csv: {cell_csv}")
    print(f"effects_csv: {effects_csv}")
    print(f"factor_coding_json: {coding_json}")
    print(f"analysis_config_json: {config_json}")
    print(f"report_txt: {report_txt}")
    for path in plot_paths:
        print(f"plot: {path}")
    for warning in warnings:
        print(f"warning: {warning}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze 2^k r factorial design results using average discounted return "
            "as the response. Each trial/seed contributes one response value."
        )
    )
    parser.add_argument(
        "sweeps",
        nargs="+",
        type=Path,
        help="Sweep directory or sweep_manifest.yaml path. Pass one per design point.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=None,
        help=(
            "CSV/TSV/JSON/YAML design table. It should contain one row per design "
            "point and factor columns with two levels."
        ),
    )
    parser.add_argument(
        "--join-column",
        default=None,
        help=(
            "Column used to join sweep responses to the design table. If omitted, "
            "the script tries experiment_number, experiment, design_id, sweep_id, "
            "source_label, and workflow columns."
        ),
    )
    parser.add_argument(
        "--factor",
        action="append",
        default=[],
        help=(
            "Factor column name. Repeat or comma-separate values. If omitted, all "
            "two-level design columns are used."
        ),
    )
    parser.add_argument("--history", choices=("train", "eval"), default="train")
    parser.add_argument("--response", default="discounted_return")
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Average only the last N rows of each trial history.",
    )
    parser.add_argument(
        "--fallback-to-return",
        action="store_true",
        help="Use return when --response discounted_return is missing.",
    )
    parser.add_argument(
        "--max-order",
        default="all",
        help="Maximum interaction order to include, or 'all'. Default: all.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/analysis/factorial_design"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.last_n is not None and args.last_n < 1:
        parser.error("--last-n must be at least 1")
    return args


def load_trial_responses(
    sweep_paths: Sequence[Path],
    *,
    history: str,
    response: str,
    last_n: int | None,
    fallback_to_return: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sweep_path in sweep_paths:
        compilation = load_sweep_manifest(sweep_path)
        histories = load_histories(sweep_path, history=history)
        if histories.empty:
            continue

        source_response = response
        response_missing = source_response not in histories.columns
        response_empty = (
            not response_missing
            and pd.to_numeric(histories[source_response], errors="coerce").dropna().empty
        )
        if response_missing or response_empty:
            if response == "discounted_return" and fallback_to_return and "return" in histories.columns:
                source_response = "return"
            else:
                raise SystemExit(
                    f"{response!r} not found in {history} histories for {sweep_path}"
                )

        trial_by_id = {trial.trial_id: trial for trial in compilation.trials}
        experiment_label, experiment_number = extract_experiment(compilation.sweep_id)
        if experiment_label is None:
            experiment_label, experiment_number = extract_experiment(compilation.name)
        if experiment_label is None:
            experiment_label, experiment_number = extract_experiment(str(sweep_path))

        for trial_id, trial_history in histories.groupby("trial_id", sort=False):
            values = response_values(trial_history, source_response, last_n=last_n)
            if values.empty:
                continue
            trial = trial_by_id.get(str(trial_id))
            first = trial_history.iloc[0]
            parameters = first.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            group_parameters = non_seed_parameters(parameters)

            row: dict[str, Any] = {
                "trial_id": str(trial_id),
                "group_id": value_or_none(first.get("group_id")),
                "seed_value": value_or_none(first.get("seed_value")),
                "run_dir": value_or_none(first.get("run_dir")),
                "sweep_id": compilation.sweep_id,
                "sweep_name": compilation.name,
                "sweep_dir": compilation.sweep_dir,
                "manifest_path": compilation.manifest_path,
                "source_label": source_label(compilation, sweep_path),
                "experiment": experiment_label,
                "experiment_number": experiment_number,
                RESPONSE_COLUMN: float(values.mean()),
                "response_count": int(len(values)),
                "response_source": source_response,
                "group_key": json.dumps(group_parameters, sort_keys=True, default=str),
            }
            if trial is not None:
                row["workflow_path"] = trial.workflow_path
                row["experiment_id"] = trial.experiment_id
            row.update(parameter_columns(group_parameters, existing=row.keys()))
            rows.append(row)

    return pd.DataFrame(rows)


def response_values(
    trial_history: pd.DataFrame,
    response: str,
    *,
    last_n: int | None,
) -> pd.Series:
    sort_columns = [column for column in ("episode", "env_step") if column in trial_history.columns]
    ordered = trial_history.sort_values(sort_columns, kind="stable") if sort_columns else trial_history
    values = pd.to_numeric(ordered[response], errors="coerce").dropna()
    if last_n is not None:
        values = values.tail(last_n)
    return values


def source_label(compilation: Any, sweep_path: Path) -> str:
    experiment_label, _experiment_number = extract_experiment(compilation.sweep_id)
    if experiment_label is not None:
        return experiment_label
    experiment_label, _experiment_number = extract_experiment(compilation.name)
    if experiment_label is not None:
        return experiment_label
    if compilation.sweep_id:
        return str(compilation.sweep_id)
    path = sweep_path.parent if sweep_path.name == "sweep_manifest.yaml" else sweep_path
    return path.name


def extract_experiment(value: Any) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    match = re.search(r"experiment[-_\s]*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        return None, None
    number = int(match.group(1))
    return f"experiment{number}", number


def parameter_columns(parameters: dict[str, Any], *, existing: Sequence[str]) -> dict[str, Any]:
    existing_set = set(existing)
    columns: dict[str, Any] = {}
    for key, value in parameters.items():
        column = str(key)
        if column in existing_set or column in columns:
            column = f"param.{column}"
        columns[column] = value
    return columns


def load_design_table(path: Path | None) -> tuple[pd.DataFrame | None, str | None]:
    if path is None:
        if not EMBEDDED_DESIGN:
            return None, None
        return augment_design_columns(pd.DataFrame(EMBEDDED_DESIGN)), "EMBEDDED_DESIGN"

    path = path.expanduser()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = raw.get("design", raw) if isinstance(raw, dict) else raw
        design = pd.DataFrame(rows)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("design", raw) if isinstance(raw, dict) else raw
        design = pd.DataFrame(rows)
    else:
        sep = "\t" if suffix in {".tsv", ".tab"} else None
        design = pd.read_csv(path, sep=sep, engine="python")

    design.columns = [str(column).strip() for column in design.columns]
    if design.empty:
        raise SystemExit(f"Design table has no rows: {path}")
    return augment_design_columns(design), str(path)


def augment_design_columns(design: pd.DataFrame) -> pd.DataFrame:
    out = design.copy()
    if "experiment" in out.columns and "experiment_number" not in out.columns:
        numbers: list[int | None] = []
        labels: list[str | None] = []
        for value in out["experiment"]:
            label, number = extract_experiment(value)
            if label is None and pd.notna(value):
                try:
                    number = int(value)
                    label = f"experiment{number}"
                except (TypeError, ValueError):
                    pass
            labels.append(label)
            numbers.append(number)
        if any(number is not None for number in numbers):
            out["experiment_number"] = numbers
        if any(label is not None for label in labels):
            out["experiment"] = [
                label if label is not None else value
                for label, value in zip(labels, out["experiment"], strict=True)
            ]
    return out


def merge_design(
    responses: pd.DataFrame,
    design: pd.DataFrame | None,
    *,
    join_column: str | None,
) -> pd.DataFrame:
    if design is None:
        return responses

    if join_column is not None:
        if join_column not in responses.columns:
            raise SystemExit(f"--join-column {join_column!r} is not in response data")
        if join_column not in design.columns:
            raise SystemExit(f"--join-column {join_column!r} is not in design table")
        try:
            return merge_on_column(responses, design, join_column)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    for candidate in JOIN_PRIORITY:
        if candidate in responses.columns and candidate in design.columns:
            try:
                return merge_on_column(responses, design, candidate)
            except ValueError:
                continue

    response_columns = ", ".join(responses.columns)
    design_columns = ", ".join(design.columns)
    raise SystemExit(
        "Could not join design table to responses. Pass --join-column. "
        f"Response columns: {response_columns}. Design columns: {design_columns}."
    )


def merge_on_column(
    responses: pd.DataFrame,
    design: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    left = responses.copy()
    right = design.copy()
    join_key = "__design_join_key"
    left[join_key] = left[column].map(normalize_join_value)
    right[join_key] = right[column].map(normalize_join_value)

    duplicates = right[right[join_key].duplicated(keep=False)][column].dropna().unique()
    if len(duplicates):
        raise ValueError(f"Design join column {column!r} has duplicate keys: {duplicates}")

    merged = left.merge(
        right,
        on=join_key,
        how="left",
        suffixes=("", "_design"),
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] != "both", column].dropna().unique()
    if len(missing):
        raise ValueError(f"Design table is missing rows for {column!r}: {list(missing)}")

    merged = merged.drop(columns=[join_key, "_merge"])
    if f"{column}_design" in merged.columns:
        merged = merged.drop(columns=[f"{column}_design"])
    merged["design_join_column"] = column
    return merged


def normalize_join_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip().lower()
    experiment_match = re.search(r"experiment[-_\s]*(\d+)", text)
    if experiment_match is not None:
        return experiment_match.group(1)
    if "/" in text or "\\" in text:
        text = Path(text).stem
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_factor_columns(raw_values: Sequence[str]) -> list[str]:
    factors: list[str] = []
    for raw in raw_values:
        factors.extend(part.strip() for part in raw.split(",") if part.strip())
    return factors


def resolve_factor_columns(
    data: pd.DataFrame,
    *,
    requested: Sequence[str],
    design_columns: Sequence[str] | None,
) -> list[str]:
    if requested:
        missing = [factor for factor in requested if factor not in data.columns]
        if missing:
            raise SystemExit(f"Requested factor columns are missing: {missing}")
        for factor in requested:
            unique_count = data[factor].dropna().nunique()
            if unique_count != 2:
                raise SystemExit(
                    f"Factor {factor!r} must have exactly two levels, found {unique_count}"
                )
        return list(requested)

    candidate_columns = list(design_columns) if design_columns is not None else list(data.columns)
    factors: list[str] = []
    for column in candidate_columns:
        if column not in data.columns:
            continue
        if column in NON_FACTOR_COLUMNS or str(column).startswith("__"):
            continue
        if str(column).endswith("_design"):
            continue
        if data[column].dropna().nunique() == 2:
            factors.append(column)
    return factors


def code_factor_columns(
    data: pd.DataFrame,
    factor_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = data.copy()
    coding: dict[str, Any] = {}
    for factor in factor_columns:
        levels = list(pd.unique(out[factor].dropna()))
        mapping = factor_level_mapping(levels)
        coded_column = coded_factor_column(factor)
        out[coded_column] = out[factor].map(mapping)
        if out[coded_column].isna().any():
            raise SystemExit(f"Could not code all levels for factor {factor!r}")
        coding[factor] = {
            "low": json_safe(level_for_code(mapping, -1)),
            "high": json_safe(level_for_code(mapping, 1)),
            "mapping": {str(key): int(value) for key, value in mapping.items()},
        }
    return out, coding


def factor_level_mapping(levels: Sequence[Any]) -> dict[Any, int]:
    if len(levels) != 2:
        raise ValueError("A factorial factor must have exactly two levels")

    explicit: dict[Any, int] = {}
    for level in levels:
        code = explicit_level_code(level)
        if code is not None:
            explicit[level] = code
    if len(explicit) == 2 and set(explicit.values()) == {-1, 1}:
        return explicit

    ordered = sorted(levels, key=lambda value: str(value))
    return {ordered[0]: -1, ordered[1]: 1}


def explicit_level_code(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else -1
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if math.isclose(number, -1.0):
            return -1
        if math.isclose(number, 1.0):
            return 1
        if math.isclose(number, 0.0):
            return -1
    normalized = str(value).strip().lower()
    if normalized in LOW_LEVELS:
        return -1
    if normalized in HIGH_LEVELS:
        return 1
    return None


def level_for_code(mapping: dict[Any, int], code: int) -> Any:
    for level, mapped_code in mapping.items():
        if mapped_code == code:
            return level
    return None


def coded_factor_column(factor: str) -> str:
    return f"__coded__{factor}"


def parse_max_order(raw: str, factor_count: int) -> int:
    if raw.lower() == "all":
        return factor_count
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit("--max-order must be an integer or 'all'") from exc
    if value < 1:
        raise SystemExit("--max-order must be at least 1")
    return min(value, factor_count)


def fit_factorial_model(
    data: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    max_order: int,
    response_column: str,
) -> dict[str, Any]:
    terms = factorial_terms(factor_columns, max_order=max_order)
    y = pd.to_numeric(data[response_column], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        raise SystemExit(f"{response_column} contains non-numeric values")

    x = design_matrix(data, terms)
    beta, *_rest = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    sse = float(residuals @ residuals)
    rank = int(np.linalg.matrix_rank(x))
    df_resid = int(len(y) - rank)
    mse_resid = sse / df_resid if df_resid > 0 else math.nan
    sst = float(((y - float(np.mean(y))) ** 2).sum())
    ss_model = max(0.0, sst - sse)
    r_squared = ss_model / sst if sst > 0 else math.nan

    balanced = balanced_replicate_count(data, factor_columns)
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(terms, start=1):
        reduced_x = np.delete(x, index, axis=1)
        reduced_beta, *_ = np.linalg.lstsq(reduced_x, y, rcond=None)
        reduced_residuals = y - reduced_x @ reduced_beta
        reduced_sse = float(reduced_residuals @ reduced_residuals)
        reduced_rank = int(np.linalg.matrix_rank(reduced_x))
        df_term = rank - reduced_rank
        partial_ss = max(0.0, reduced_sse - sse)
        ms_term = partial_ss / df_term if df_term > 0 else math.nan
        f_stat = ms_term / mse_resid if df_term > 0 and df_resid > 0 and mse_resid > 0 else math.nan
        contrast_effect = term_contrast_effect(data, term, response_column)
        rows.append(
            {
                "term": term_label(term),
                "order": len(term),
                "contrast_effect": contrast_effect,
                "ols_coefficient": float(beta[index]),
                "ols_effect": float(2.0 * beta[index]),
                "ss_partial": partial_ss,
                "ss_orthogonal": (
                    len(y) * contrast_effect**2 / 4.0 if balanced is not None else math.nan
                ),
                "df": df_term,
                "ms": ms_term,
                "f": f_stat,
                "p_value": f_survival(f_stat, df_term, df_resid),
            }
        )

    effects = pd.DataFrame(rows)
    if not effects.empty:
        effects = effects.sort_values(
            ["order", "contrast_effect"],
            key=lambda column: column.abs() if column.name == "contrast_effect" else column,
            ascending=[True, False],
        ).reset_index(drop=True)

    warnings = []
    if rank < x.shape[1]:
        warnings.append(
            f"Model matrix is rank deficient: rank={rank}, columns={x.shape[1]}. "
            "Some high-order effects may be aliased."
        )
    if df_resid <= 0:
        warnings.append("No residual degrees of freedom; F statistics and p-values are unavailable.")
    if balanced is None:
        warnings.append(
            "The design is not a complete balanced 2^k design. Partial sums of squares "
            "come from OLS model comparisons; orthogonal factorial SS is unavailable."
        )

    model_summary = {
        "n": int(len(y)),
        "term_count": len(terms),
        "rank": rank,
        "df_resid": df_resid,
        "sse": sse,
        "mse_resid": mse_resid,
        "sst": sst,
        "ss_model": ss_model,
        "r_squared": r_squared,
        "grand_mean": float(np.mean(y)),
        "balanced_replicates": balanced,
    }
    return {"effects": effects, "model_summary": model_summary, "warnings": warnings}


def factorial_terms(factor_columns: Sequence[str], *, max_order: int) -> list[tuple[str, ...]]:
    terms: list[tuple[str, ...]] = []
    for order in range(1, max_order + 1):
        terms.extend(itertools.combinations(factor_columns, order))
    return terms


def design_matrix(data: pd.DataFrame, terms: Sequence[tuple[str, ...]]) -> np.ndarray:
    columns = [np.ones(len(data), dtype=float)]
    for term in terms:
        term_column = np.ones(len(data), dtype=float)
        for factor in term:
            term_column *= data[coded_factor_column(factor)].to_numpy(dtype=float)
        columns.append(term_column)
    return np.column_stack(columns)


def term_contrast_effect(
    data: pd.DataFrame,
    term: Sequence[str],
    response_column: str,
) -> float:
    sign = np.ones(len(data), dtype=float)
    for factor in term:
        sign *= data[coded_factor_column(factor)].to_numpy(dtype=float)
    y = data[response_column].to_numpy(dtype=float)
    high = y[sign > 0]
    low = y[sign < 0]
    if len(high) == 0 or len(low) == 0:
        return math.nan
    return float(np.mean(high) - np.mean(low))


def balanced_replicate_count(data: pd.DataFrame, factor_columns: Sequence[str]) -> int | None:
    if not factor_columns:
        return None
    coded_columns = [coded_factor_column(factor) for factor in factor_columns]
    counts = data.groupby(coded_columns, dropna=False).size()
    if len(counts) != 2 ** len(factor_columns):
        return None
    unique_counts = counts.unique()
    if len(unique_counts) != 1:
        return None
    return int(unique_counts[0])


def f_survival(f_stat: float, df_num: int, df_den: int) -> float:
    if math.isnan(f_stat) or df_num <= 0 or df_den <= 0:
        return math.nan
    try:
        from scipy.stats import f as f_distribution
    except Exception:
        return math.nan
    return float(f_distribution.sf(f_stat, df_num, df_den))


def term_label(term: Sequence[str]) -> str:
    return ":".join(term)


def summarize_cells(data: pd.DataFrame, factor_columns: Sequence[str]) -> pd.DataFrame:
    group_columns = list(factor_columns) if factor_columns else ["source_label"]
    rows: list[dict[str, Any]] = []
    for key, group in data.groupby(group_columns, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        response = group[RESPONSE_COLUMN].to_numpy(dtype=float)
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "mean": float(np.mean(response)),
                "std": float(np.std(response, ddof=1)) if len(response) > 1 else 0.0,
                "se": (
                    float(np.std(response, ddof=1) / math.sqrt(len(response)))
                    if len(response) > 1
                    else 0.0
                ),
                "min": float(np.min(response)),
                "max": float(np.max(response)),
                "replicates": int(len(response)),
                "seed_count": int(group["seed_value"].nunique()) if "seed_value" in group else 0,
                "source_labels": sorted(map(str, group["source_label"].dropna().unique())),
                "sweep_ids": sorted(map(str, group["sweep_id"].dropna().unique())),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def format_report(
    *,
    responses: pd.DataFrame,
    cell_summary: pd.DataFrame,
    effects: pd.DataFrame,
    factor_columns: Sequence[str],
    model_summary: dict[str, Any],
    warnings: Sequence[str],
) -> str:
    lines = [
        "Factorial Design Analysis",
        "",
        f"Trials: {len(responses)}",
        f"Design cells: {len(cell_summary)}",
        f"Response: {RESPONSE_COLUMN}",
    ]
    if factor_columns:
        lines.append(f"Factors ({len(factor_columns)}): {', '.join(factor_columns)}")
    else:
        lines.append("Factors: none detected")

    if model_summary:
        balanced = model_summary.get("balanced_replicates")
        balanced_text = str(balanced) if balanced is not None else "no"
        lines.extend(
            [
                "",
                "Model",
                f"Grand mean: {format_number(model_summary.get('grand_mean'))}",
                f"R^2: {format_number(model_summary.get('r_squared'))}",
                f"Residual df: {model_summary.get('df_resid')}",
                f"Residual MSE: {format_number(model_summary.get('mse_resid'))}",
                f"Balanced replicates per cell: {balanced_text}",
            ]
        )

    if not effects.empty:
        top_effects = effects.copy()
        top_effects["abs_effect"] = top_effects["contrast_effect"].abs()
        top_effects = top_effects.sort_values("abs_effect", ascending=False).head(20)
        lines.extend(["", "Largest Effects"])
        lines.append(
            render_table(
                ["term", "order", "effect", "F", "p"],
                [
                    [
                        str(row["term"]),
                        str(int(row["order"])),
                        format_number(row["contrast_effect"]),
                        format_number(row["f"]),
                        format_number(row["p_value"]),
                    ]
                    for _, row in top_effects.iterrows()
                ],
            )
        )

    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines) + "\n"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    rendered = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        rendered.append(
            "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        )
    return "\n".join(rendered)


def write_plots(
    effects: pd.DataFrame,
    data: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    out_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[Path] = []
    if not effects.empty:
        plot_df = effects.copy()
        plot_df["abs_effect"] = plot_df["contrast_effect"].abs()
        plot_df = plot_df.sort_values("abs_effect", ascending=True).tail(30)
        height = max(3.0, 0.25 * len(plot_df) + 1.0)
        fig, ax = plt.subplots(figsize=(8.0, height), dpi=160)
        ax.barh(plot_df["term"], plot_df["abs_effect"], color="#4c78a8")
        ax.set_xlabel("Absolute contrast effect")
        ax.set_ylabel("Term")
        ax.set_title("Factorial Effects")
        fig.tight_layout()
        path = out_dir / "effects_pareto.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    main_effects = [factor for factor in factor_columns if coded_factor_column(factor) in data.columns]
    if main_effects:
        columns = min(3, len(main_effects))
        rows = math.ceil(len(main_effects) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), dpi=160)
        axes_array = np.asarray(axes).reshape(-1)
        for ax, factor in zip(axes_array, main_effects, strict=False):
            grouped = (
                data.groupby(coded_factor_column(factor))[RESPONSE_COLUMN]
                .mean()
                .reindex([-1, 1])
            )
            ax.plot(["low", "high"], grouped.to_numpy(dtype=float), marker="o", color="#f58518")
            ax.set_title(factor)
            ax.set_ylabel("Mean response")
        for ax in axes_array[len(main_effects) :]:
            ax.axis("off")
        fig.tight_layout()
        path = out_dir / "main_effects.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    export = df.copy()
    for column in export.columns:
        export[column] = export[column].map(csv_cell)
    export.to_csv(path, index=False)


def internal_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if str(column).startswith("__coded__")]


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=json_safe)
    return json_safe(value)


def value_or_none(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float, np.integer, np.floating)) or math.isnan(float(value)):
        return "-"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
