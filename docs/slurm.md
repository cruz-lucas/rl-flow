# SLURM

SLURM execution is configured through `WorkflowSpec.execution.options` or cluster YAML files such as `configs/clusters/slurm.example.yaml`.

Supported options:

- `partition`
- `account`
- `time`
- `nodes`
- `ntasks`
- `cpus_per_task`
- `mem`
- `gres`
- `constraint`
- `qos`
- `reservation`
- `mail_user`
- `mail_type`
- `modules`
- `venv_path`
- `conda_env`
- `preamble`
- `env`

The compiler writes `slurm_job.sh` when the workflow backend is `slurm`. `SlurmExecutor.submit` calls `sbatch` if it is available. Status checks use `squeue` first and fall back to `sacct`.

Local development does not require SLURM; rendering is covered by tests without calling `sbatch`.

## Sweep Arrays

Hyperparameter sweeps use SLURM job arrays. Define a sweep file with a base workflow, search parameters, and SLURM options:

```yaml
name: navix_dqn_hparam_sweep
workflow: ../workflows/navix_dqn_empty_room.yaml
method: grid
execution:
  backend: slurm
  options:
    account: def-yourpi
    time: "02:00:00"
    cpus_per_task: 4
    mem: 16G
    gres: gpu:1
slurm:
  max_parallel: 8
parameters:
  learning_rate:
    target: nodes.agent.config.learning_rate
    values: [0.001, 0.0003, 0.0001]
  batch_size:
    target: nodes.replay.config.batch_size
    values: [32, 64]
```

Compile or submit it from the cluster checkout of this repository so generated `command.sh` files point at the correct project root:

```bash
uv run python -m rlflow.cli sweep compile configs/sweeps/navix_dqn_compute_canada.yaml --out runs/sweeps/navix-dqn
uv run python -m rlflow.cli sweep run configs/sweeps/navix_dqn_compute_canada.yaml --out runs/sweeps/navix-dqn
```

Compilation writes:

- `sweep_manifest.yaml`
- `slurm_array.sh`
- one `trials/trial-*/` run directory per hyperparameter assignment

Each target path starts with `nodes.<node_id>.config`, for example `nodes.agent.config.learning_rate`. `method: random` also supports `values`, `uniform`, `loguniform`, and `int_uniform` parameters with `num_trials` and `seed`.

Summarize completed trials by reading each trial `metrics.json`:

```bash
uv run python -m rlflow.cli sweep summarize runs/sweeps/navix-dqn
```
