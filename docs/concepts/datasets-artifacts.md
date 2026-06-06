# Datasets and Artifacts

`rl-flow` can save replay datasets, inspect `.npz` transition datasets, and run offline intrinsic-reward analysis through the API and UI.

## Transition Dataset Shape

The dataset inspector expects compressed `.npz` files. A transition dataset should include:

- `observations`
- `actions`
- `rewards`
- `next_observations`
- `terminals`

Optional arrays, such as `cfn_targets`, can support specific intrinsic-reward analyses.

## Dataset API

Use the API to list and inspect datasets:

```bash
curl http://127.0.0.1:8000/datasets
```

Inspect a dataset:

```bash
curl -X POST http://127.0.0.1:8000/datasets/inspect \
  -H 'Content-Type: application/json' \
  -d '{"path": "runs/example/artifacts/replay/dataset.npz"}'
```

## Offline Intrinsic-Reward Analysis

The offline endpoint can compare count-based and learned bonuses for algorithms such as RND, CFN, classifier-style novelty, and SimHash.

This is useful for debugging exploration bonuses before committing to a full training run.

## Artifact API

Experiment artifacts are listed by experiment ID:

```bash
curl http://127.0.0.1:8000/artifacts/{experiment_id}
```

The current artifact browser is intentionally minimal. A research-grade system should add typed artifact metadata, preview renderers, result cards, and stable artifact URIs.
