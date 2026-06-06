# Navix DQN

The Navix DQN workflow exercises vector or symbolic observations, replay, a neural Q-network, and the built-in JAX runner. The concrete example is:

```text
configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn.yaml
```

## Validate and Compile

```bash
uv run rlflow workflow validate configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn.yaml
uv run rlflow compile configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn.yaml --out runs/examples/navix-dqn
```

## Run

```bash
uv run rlflow run configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn.yaml --backend local --out runs/examples/navix-dqn-local
```

This workflow connects:

- `builtin.agent.dqn_jax`
- `navix.env.grid`
- `builtin.replay.uniform`
- `builtin.runner.tabular_jax`

The DQN path supports observation normalization, configurable hidden layers, optimizer settings, target network updates, double Q-learning, Huber or MSE losses, and replay updates.

## Why This Example Matters

The `symbolic` observation mode and `corner_wall_color` distractor make this useful for checking whether representation and exploration choices behave as expected. It is also the baseline for comparing the DQN + R-Max count-oracle workflow.

## Expected Research Artifacts

After a successful run, inspect:

- `logs/train_history.jsonl` for per-episode training returns.
- `logs/eval_history.jsonl` when evaluation episodes are enabled.
- `summaries/metrics.json` for scalar summary metrics.
- `artifacts/replay/` when replay export is configured.
- `manifest.json` for provenance and file hashes.
