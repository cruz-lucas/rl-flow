# SLURM Sweeps

The SLURM sweep example is:

```text
configs/sweeps/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_countbasedoracle.yaml
```

It points at the DQN + R-Max count-oracle workflow, expands a grid of learning rates, replay sizes, target update periods, normalization settings, R-Max thresholds, and seeds, then batches trials through a SLURM array.

## Compile

```bash
uv run rlflow sweep compile configs/sweeps/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_countbasedoracle.yaml --out runs/sweeps/navix-dqn
```

Compilation writes:

- `sweep_manifest.yaml`
- `slurm_array.sh`
- one trial run directory per hyperparameter assignment
- one `workflow.yaml`, `resolved_config.yaml`, `generated.gin`, `command.sh`, and `manifest.json` per trial

## Submit

Run from the cluster checkout so generated commands point at the right project root:

```bash
uv run rlflow sweep run configs/sweeps/16x16_emptyroom_symbolic_cardinal_cornerdistractor__dqn_countbasedoracle.yaml --out runs/sweeps/navix-dqn
```

The sweep config controls queue pressure:

```yaml
slurm:
  max_parallel: 8
  trials_per_task: 50
  max_array_tasks: 1000
```

`trials_per_task` runs several trials serially inside one array element. Set `execution.options.time` for the whole serial batch, not a single trial.

## Track Progress

```bash
uv run rlflow sweep status runs/sweeps/navix-dqn
```

The status command reads trial filesystem state and prints counts for completed, failed, running, queued, compiled, cancelled, unknown, and missing trials.

## Summarize

```bash
uv run rlflow sweep summarize runs/sweeps/navix-dqn --metric mean_train_return_last_n --metric-last-n 500
uv run rlflow sweep report runs/sweeps/navix-dqn --metric mean_train_return_last_n --metric-last-n 500 --out runs/sweeps/navix-dqn/analysis
```

Seed trials for the same non-seed hyperparameters are grouped before ranking.
