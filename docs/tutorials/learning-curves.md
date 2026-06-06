# Learning-Curve Analysis

Sweep histories are JSONL files under each trial's `logs/` directory. The analysis commands load those histories, group seed replicates, interpolate curves onto common x-values, compute aggregate bands, and export CSV/SVG outputs.

## Export Curves

```bash
uv run rlflow sweep export-learning-curves runs/sweeps/navix-dqn --history train --value discounted_return
```

This writes seed-averaged discounted-return curves with bootstrapped confidence bands.

## Plot Curves

```bash
uv run rlflow sweep plot-learning-curves runs/sweeps/navix-dqn \
  --history train \
  --x env_step \
  --y discounted_return \
  --top-k 5 \
  --sort-by mean_train_discounted_return_last_n \
  --goal maximize \
  --smooth-window 5
```

The command writes raw histories, interpolated curves, plot-ready curves when smoothing is enabled, a resolved plot config, and figure outputs.

## Reuse an Existing Result

The repository currently includes a generated example artifact directory:

```text
16x16-emptyroom-symbolic-cardinal-corner-dqn--best/learning_curves/
```

Use the CSV and SVG there as an example of the expected analysis shape. Treat this as an artifact snapshot, not as a source of truth for current CLI behavior.

## Research Reporting Guidance

For a paper-style result, report:

- sweep manifest path and git commit
- metric and aggregation window
- seed count per group
- mean, min, max, and standard deviation across seeds
- curve x-axis and interpolation settings
- confidence-band method and bootstrap sample count
