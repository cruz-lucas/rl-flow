# Sweep Schema

Sweep schemas are Pydantic models in `rlflow.schemas.sweep`.

## `SweepSpec`

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Sweep name. |
| `description` | string | Optional description. |
| `sweep_id` | string or null | Optional stable sweep ID. |
| `workflow` | path or `WorkflowSpec` | Base workflow. |
| `method` | `grid` or `random` | Search strategy. |
| `metric` | `SweepMetric` | Metric used for ranking. |
| `parameters` | map of `SweepParameter` | Search dimensions. |
| `num_trials` | integer or null | Random-search non-seed assignment count. |
| `seed` | integer | Random-search seed. |
| `execution` | `ExecutionSpec` or null | Optional backend override. |
| `slurm` | `SweepSlurmSpec` | SLURM array controls. |
| `metadata` | object | Free-form metadata. |

## `SweepParameter`

| Field | Type | Description |
| --- | --- | --- |
| `target` | string | Workflow config path, such as `nodes.agent.config.learning_rate`. |
| `values` | list or null | Explicit values for grid or choice sampling. |
| `distribution` | string | `choice`, `uniform`, `loguniform`, or `int_uniform`. |
| `minimum` | number or null | Lower bound for sampled distributions. |
| `maximum` | number or null | Upper bound for sampled distributions. |

## `SweepMetric`

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Metric name, default `mean_eval_return`. |
| `goal` | `maximize` or `minimize` | Ranking direction. |
| `last_n` | integer or null | Optional history window. |

## `SweepSlurmSpec`

| Field | Type | Description |
| --- | --- | --- |
| `max_parallel` | integer or null | Maximum concurrent array tasks. |
| `trials_per_task` | integer | Serial trials per array task. |
| `max_array_tasks` | integer or null | Fail early if the array would exceed this. |

## Generated Manifest

Compilation writes `SweepCompilation`, including the sweep ID, manifest path, SLURM array path, generated files, and every `SweepTrial`.
