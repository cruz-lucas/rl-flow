# Factorial Design Analysis

This tutorial explains how to design, run, analyze, and report a replicated two-level
factorial experiment with `scripts/analyze_factorial_design.py`. It also explains what
the generated effects, sums of squares, F tests, p-values, Welch ANOVA, and variation
percentages mean.

## What the Script Analyzes

The script is intended for a replicated `2^k` full factorial design:

- `k` is the number of factors.
- Every factor has exactly two levels, conventionally coded `-1` and `+1`.
- A full design contains all `2^k` combinations of those levels.
- Each combination, or **design cell**, is repeated with independent trials or seeds.

For example, three two-level factors produce eight cells. With 30 seeds per cell, the
analysis has `8 * 30 = 240` trial responses. See the
[NIST full-factorial overview](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)
for the general design structure.

The experimental unit in this analysis is one trial or seed, not one episode. For each
trial, the script averages the selected history metric and produces one value named
`average_discounted_return`. Episodes from the same trial are correlated observations
and must not be counted as independent replicates.

## Theory

### Factors, cells, and replication

A factor is a controlled experimental choice such as replay type, optimism setting, or
whether Double Q-learning is enabled. Each factor must have a low and high level. A
full factorial experiment evaluates every combination, allowing main effects and
interactions to be estimated separately.

Replication provides the within-cell variability needed to estimate experimental
noise. Without replication, a model containing every factorial interaction has no
residual degrees of freedom, so its F tests and p-values are unavailable.

### Coded-factor model

For two factors `A` and `B`, the fitted model is:

```text
y = beta_0 + beta_A*x_A + beta_B*x_B + beta_AB*x_A*x_B + error
```

Each `x` is `-1` at the low level and `+1` at the high level. Additional factors add
their main-effect columns and products such as `x_A*x_C` and `x_A*x_B*x_C`.

In a complete balanced design:

- `beta_0` is the grand mean.
- `2 * beta_A` is the mean response at high `A` minus the mean response at low `A`.
- An interaction measures whether a factor's effect changes with another factor's
  level.
- The coded model columns are orthogonal, so each term has an independent sum of
  squares.

The script calls `2 * beta` the `ols_effect`. It also calculates a
`contrast_effect`: the mean where a term's coded product is `+1` minus the mean where
it is `-1`. These are equal in a complete balanced design. For a two-factor
interaction, this value is one half of the usual difference-in-differences because of
the script's `-1`/`+1` effect scaling.

### Main effects and interactions

A main effect averages over the levels of every other factor. A positive main effect
means changing that factor from low to high increased the response on average. For a
return metric, that is usually favorable; for a cost metric, the preferred sign may be
the opposite.

An interaction means the main-effect average is incomplete on its own. If `A:B` is
large, inspect the cell means before claiming that `A` is uniformly beneficial. It may
help when `B` is high and hurt when `B` is low.

Use the hierarchical principle when choosing a model: if an interaction is included,
retain its lower-order component terms even when their individual p-values are large.
The default `--max-order all` follows this principle for a full factorial model.

### Sums of squares and ANOVA

Total response variation is measured by:

```text
SS_total = sum((y_i - grand_mean)^2)
```

For a complete balanced design fitted through all interaction orders:

```text
SS_total = sum(SS_each_factor_and_interaction) + SS_residual
```

The script reports two term sums of squares:

- `ss_orthogonal` is the factorial sum of squares. It is available only when all
  `2^k` cells have the same replicate count.
- `ss_partial` is the increase in residual sum of squares after removing that term
  from the fitted OLS model. This remains available for unbalanced data, but partial
  sums of squares are not generally additive.

For each term, the ordinary factorial F test is:

```text
F = MS_term / MSE_residual
```

Its p-value tests the null hypothesis that the corresponding model term is zero,
assuming independent observations, an adequate linear factorial model, approximately
normal residual errors, and a common residual variance. A p-value is evidence about
sampling uncertainty, not the practical size or importance of an effect.

### Percentage of variation explained

For a complete balanced design, the percentage of total response variation assigned
to a term is:

```text
variation_percent = 100 * ss_orthogonal / SS_total
```

Main factors are rows where `order == 1`; higher orders are interactions. When all
orders are fitted, the factor, interaction, and residual percentages sum to 100%,
apart from floating-point rounding.

This is different from the percentage of **model-explained** variation. Dividing a
term's sum of squares by the sum across model terms excludes residual noise and makes
the modeled terms sum to 100%. State explicitly which denominator you use.

For an unbalanced design, do not present `ss_partial / SS_total` values as additive
shares. A common alternative is partial eta squared:

```text
partial_eta_squared = ss_partial / (ss_partial + SS_residual)
```

Partial eta-squared values answer a different question and do not sum to 100%.

### Welch ANOVA

`--welch-anova` adds a one-way Welch omnibus test across the observed design cells. Its
null hypothesis is that every cell has the same population mean. The test weights cell
means using their sample sizes and variances and uses an adjusted denominator degrees
of freedom. It does not require equal cell variances.

Welch ANOVA does **not** provide separate tests or variation percentages for main
effects and interactions. It answers only whether there is evidence that at least one
cell mean differs. The existing OLS factorial results remain in `effects.csv`.

The implementation requires at least two observed cells, at least two responses per
cell, and positive within-cell variance. Independence and approximate within-cell
normality remain assumptions. The
[SciPy one-way ANOVA documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
provides additional background on the standard and Welch variants.

## Step 1: Install Analysis Dependencies

From the repository root:

```bash
uv sync --extra analysis
```

Use `uv run` for the remaining commands so the script runs in the managed environment.

## Step 2: Define the Design Before Inspecting Results

Create a CSV with one row per design cell. The following is an illustrative three-factor
design; replace the factor names and verify that each row matches the actual workflow
used for that experiment.

```csv
experiment,replay,rmax_optimism,double_q
1,-1,-1,-1
2,1,-1,-1
3,-1,1,-1
4,1,1,-1
5,-1,-1,1
6,1,-1,1
7,-1,1,1
8,1,1,1
```

Save it as, for example, `factorial_design.csv`. The `experiment` values need to match
the numbers in sweep names such as `experiment1_30_seeds`.

The script also accepts TSV, JSON, and YAML design tables. An `EMBEDDED_DESIGN` can be
defined in the script while iterating, but a separate version-controlled design file is
preferred because it records the experimental mapping without changing analysis code.

The safest factor levels are explicit `-1` and `+1`. The script also recognizes common
values such as `low`/`high`, `false`/`true`, and `off`/`on`. Other two-level values are
sorted as strings to choose low and high, so always inspect `factor_coding.json`.

## Step 3: Confirm Sweep Inputs

Each positional input must be a sweep directory or its `sweep_manifest.yaml`. Confirm
that each cell has the intended independent seeds and history files. For the repository's
factorial sweep layout, list them with:

```bash
find runs/sweeps/factorial_design_experiments \
  -maxdepth 2 -name sweep_manifest.yaml -print | sort
```

The script extracts labels such as `experiment1` from the sweep ID, sweep name, or path.
Passing `--join-column experiment_number` then joins those responses to the design rows
independently of argument order.

## Step 4: Choose the Trial Response

The default response source is `discounted_return` from the training history. For each
trial, the script averages all available values:

```text
trial response = mean(discounted_return values within that trial)
```

Use `--last-n N` to average only the last `N` history rows after sorting by episode or
environment step. This is useful for steady-state performance, but choose `N` before
looking at results and report it.

Other choices include:

```bash
--history eval
--response return
--last-n 100
```

`--fallback-to-return` uses `return` when `discounted_return` is unavailable. Check the
`response_source` column in `trial_responses.csv` to ensure different cells were not
accidentally analyzed with different response definitions.

## Step 5: Run the Analysis

An explicit invocation for three factors is:

```bash
uv run python scripts/analyze_factorial_design.py \
  runs/sweeps/factorial_design_experiments/experiment*_30_seeds \
  --design factorial_design.csv \
  --join-column experiment_number \
  --factor replay \
  --factor rmax_optimism \
  --factor double_q \
  --history train \
  --response discounted_return \
  --last-n 100 \
  --max-order all \
  --welch-anova \
  --out runs/analysis/factorial-design
```

Repeat `--factor` or pass a comma-separated list. Explicit factors are recommended. If
they are omitted, the script treats every eligible two-level design column as a factor,
which can accidentally include two-valued metadata.

Omit `--welch-anova` when only the ordinary factorial model is required. Omit
`--last-n` when the response should average the entire trial history. Use `--no-plots`
for CSV and text output only.

## Step 6: Check the Generated Files

The output directory contains:

| File | Purpose |
| --- | --- |
| `analysis_config.json` | Resolved inputs and analysis settings. |
| `factor_coding.json` | Actual low/high mapping used for every factor. |
| `trial_responses.csv` | One response per trial or seed plus provenance. |
| `cell_summary.csv` | Cell means, standard deviations, standard errors, ranges, and replicate counts. |
| `effects.csv` | Main effects, interactions, sums of squares, F statistics, and p-values. |
| `factorial_report.txt` | Human-readable model and effect summary. |
| `effects_pareto.png` | Largest absolute factorial contrasts. |
| `main_effects.png` | Mean response at the low and high level of each factor. |
| `welch_anova.json` | Welch omnibus result; written only with `--welch-anova`. |

Check these items before interpreting significance:

1. `factor_coding.json` assigns the intended low and high levels.
2. Every expected cell appears in `cell_summary.csv`.
3. `replicates` is constant across cells if the design is intended to be balanced.
4. `response_source` is consistent in `trial_responses.csv`.
5. The report contains no rank-deficiency or missing-residual-degrees-of-freedom warning.

## Step 7: Interpret `effects.csv`

The important columns are:

| Column | Interpretation |
| --- | --- |
| `term` | Factor name or colon-separated interaction, such as `replay:rmax_optimism`. |
| `order` | `1` for main effects, `2` for two-factor interactions, and so on. |
| `contrast_effect` | High-minus-low coded contrast; use its sign and magnitude. |
| `ols_coefficient` | Fitted coded-regression coefficient. |
| `ols_effect` | Twice the OLS coefficient, on the script's effect scale. |
| `ss_partial` | Extra model sum of squares attributable to the term conditional on the others. |
| `ss_orthogonal` | Orthogonal factorial sum of squares for a complete balanced design. |
| `df`, `ms` | Term degrees of freedom and mean square. |
| `f`, `p_value` | Ordinary OLS factorial F test and p-value. |

Read interactions before settling on a main-effect conclusion. Use `cell_summary.csv` to
identify which factor combinations produce the interaction and whether the best mean is
also acceptably variable.

## Step 8: Calculate Variation Percentages

For a complete balanced design, run the following from the repository root:

```bash
uv run python - <<'PY'
from pathlib import Path

import pandas as pd

out = Path("runs/analysis/factorial-design")
effects = pd.read_csv(out / "effects.csv")
responses = pd.read_csv(out / "trial_responses.csv")
y = responses["average_discounted_return"]

if effects["ss_orthogonal"].isna().any():
    raise SystemExit("Variation percentages require a complete balanced design")

ss_total = ((y - y.mean()) ** 2).sum()
effects["variation_percent"] = 100 * effects["ss_orthogonal"] / ss_total

columns = ["term", "order", "contrast_effect", "p_value", "variation_percent"]
print(effects[columns].sort_values("variation_percent", ascending=False).to_string(index=False))

residual_percent = 100 - effects["variation_percent"].sum()
print(f"\nResidual variation: {residual_percent:.3f}%")

effects.to_csv(out / "effects_with_variation.csv", index=False)
PY
```

To show only individual factors, filter with:

```python
main_effects = effects.loc[effects["order"] == 1]
```

The residual calculation above represents pure residual variation only when the model
contains every factorial order. If `--max-order` omitted higher-order terms, it also
contains variation associated with those omitted terms.

## Step 9: Interpret the Welch Result

When enabled, `welch_anova.json` contains:

- `group_count` and `observation_count`
- the factors defining each design cell
- `f_statistic`
- numerator and adjusted denominator degrees of freedom
- `p_value`

A small Welch p-value supports the conclusion that the cell means are not all equal. It
does not say which cells differ, which factors explain the difference, or whether the
difference is practically important. Use the factorial effects and cell summaries for
those questions. The script does not currently perform post-hoc pairwise comparisons.

## Assumptions and Common Failure Modes

### Independence

Seeds should represent independent trials. If configurations reuse coupled randomness
or deliberately use common random numbers, the observations may be paired; this script
treats them as independent and does not exploit that pairing.

### Balanced versus unbalanced data

Missing or failed seeds make a design unbalanced. OLS coefficients and partial sums of
squares can still be produced when the model is estimable, but orthogonal sums of
squares and additive variation percentages are unavailable.

### Variance and distribution assumptions

Ordinary term F tests use a common residual variance. Compare the cell standard
deviations and inspect trial-level responses for extreme skew or outliers. Welch ANOVA
relaxes equal variance for the omnibus cell comparison, but it does not repair dependent
observations, an incorrectly defined response, or a missing design cell.

### Rank deficiency and aliasing

A rank-deficient warning means some requested effects cannot be estimated separately.
This usually indicates missing cells, duplicated/incorrect factor columns, or a model
that asks for more terms than the observed design supports. Do not interpret aliased
term p-values.

### Joining failures

If the design table cannot be joined automatically, pass `--join-column` explicitly.
The selected column must exist in both the loaded response metadata and design table,
and the design keys must be unique.

### No residual degrees of freedom

One response per cell with all interactions fitted creates a saturated model. Add
independent replication rather than dropping interactions solely to manufacture an
error estimate.

### Multiple testing

Testing many main effects and interactions increases the chance of at least one small
p-value under the global null. Pre-specify primary effects, report all tested terms, and
consider multiplicity control when confirmatory conclusions depend on several p-values.

## Reporting Checklist

For a reproducible result, report:

- factor names and the real settings represented by low and high
- complete design table and whether the design was balanced
- trial count and seed count per cell
- history source, response column, and `--last-n` choice
- model interaction order
- effect estimates with signs and units
- sums of squares or variation percentages
- F statistics, degrees of freedom, and p-values
- whether the ordinary factorial test, Welch omnibus test, or both were used
- any failed or excluded trials
- sweep manifests, analysis configuration, and git commit

Keep practical and statistical conclusions separate. A small effect can be statistically
detectable with many seeds, while a large but noisy effect can remain uncertain.
