# RiverSwim Q-Learning

This tutorial is the smallest reproducible experiment in the repository. It uses `configs/workflows/tabular_q_learning_riverswim.yaml`, a tabular environment, a tabular Q-learning agent, an epsilon-greedy policy, and the built-in JAX runner.

## Validate

```bash
uv run rlflow workflow validate configs/workflows/tabular_q_learning_riverswim.yaml
```

Expected result:

```text
Workflow is valid
```

## Compile

```bash
uv run rlflow compile configs/workflows/tabular_q_learning_riverswim.yaml --out runs/examples/riverswim
```

Compilation writes the reproducible run files before any training starts. This is useful when you want to inspect exactly what will run on a cluster.

## Run Locally

```bash
uv run rlflow run configs/workflows/tabular_q_learning_riverswim.yaml --backend local --out runs/examples/riverswim-local
```

The run directory should include:

```text
runs/examples/riverswim-local/
  workflow.yaml
  resolved_config.yaml
  generated.gin
  command.sh
  manifest.json
  status.json
  logs/
  summaries/
  artifacts/
```

## Inspect Outputs

Check status through the CLI:

```bash
uv run rlflow jobs list
```

Look at training history and metrics:

```bash
ls runs/examples/riverswim-local/logs
ls runs/examples/riverswim-local/summaries
```

For research use, the important property is that `manifest.json` records the run ID, git commit, dirty state, dependency versions, platform, backend, and hashes for the generated files.
