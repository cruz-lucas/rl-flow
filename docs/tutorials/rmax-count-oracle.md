# DQN + R-Max Count Oracle

The intrinsic-reward example is:

```text
configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_rmax_countbasedoracle.yaml
```

It connects `builtin.agent.dqn_rmax_jax` to `builtin.intrinsic.count`. The intrinsic module acts as an unknown-state detector. Unknown actions can receive optimistic R-Max values during action selection and target updates.

## Validate

```bash
uv run rlflow workflow validate configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_rmax_countbasedoracle.yaml
```

The validator requires DQN-style agents to use `builtin.replay.uniform` on the runner `replay_buffer` port. It also requires `builtin.agent.dqn_rmax_jax` to have an `intrinsic_reward` input.

## Run Locally

```bash
uv run rlflow run configs/workflows/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_rmax_countbasedoracle.yaml --backend local --out runs/examples/rmax-count-oracle
```

## Key Configuration Ideas

- `rmax_bonus_threshold` controls when an action is treated as unknown.
- `rmax_decision_v_max` controls optimistic action selection.
- `rmax_update_v_max` controls optimistic bootstrapping in replay updates.
- `count_key_mode: oracle_tabular` uses environment state identity when available.
- `count_action_conditioning: input` makes the count bonus state-action aware.

## Comparison Protocol

Use this workflow beside the plain DQN workflow with matched environment, runner horizon, and seeds. For sweep comparison, group trials by non-seed hyperparameters and rank by `mean_train_return_last_n` or `mean_train_discounted_return_last_n`.
