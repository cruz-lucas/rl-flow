export type ComponentKind =
  | "agent"
  | "environment"
  | "runner"
  | "policy"
  | "replay_buffer"
  | "network"
  | "intrinsic_reward"
  | "logger"
  | "sweeper"
  | "launcher"
  | "analysis";

export interface PortSpec {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

export interface ComponentSpec {
  id: string;
  source: string;
  version: string;
  kind: ComponentKind;
  display_name: string;
  description: string;
  input_ports: PortSpec[];
  output_ports: PortSpec[];
  config_schema: JsonSchema;
  defaults: Record<string, unknown>;
  compile_target: Record<string, unknown>;
}

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
  enum?: unknown[];
  default?: unknown;
  minimum?: number;
  exclusiveMinimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  additionalProperties?: boolean;
  deprecated?: boolean;
  "x-inspector-hidden"?: boolean;
}

export interface WorkflowNodeSpec {
  id: string;
  component: string;
  config: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface WorkflowEdgeSpec {
  from_node: string;
  from_port: string;
  to_node: string;
  to_port: string;
}

export interface WorkflowSpec {
  name: string;
  description: string;
  nodes: WorkflowNodeSpec[];
  edges: WorkflowEdgeSpec[];
  execution: {
    backend: "local" | "slurm";
    cluster?: string | null;
    options: Record<string, unknown>;
  };
  metadata: Record<string, unknown>;
}

export interface WorkflowGalleryItem {
  workflow_id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface ValidationErrorDetail {
  message: string;
  node_id?: string | null;
  field?: string | null;
  code: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationErrorDetail[];
}

export interface ExperimentSpec {
  experiment_id: string;
  workflow: WorkflowSpec;
  resolved_config: Record<string, Record<string, unknown>>;
  run_dir: string;
  command: string;
  generated_files: string[];
  execution_backend: "local" | "slurm";
}

export interface JobInfo {
  job_id: string;
  experiment_id: string;
  backend: string;
  status: { state: string; message: string };
  run_dir: string;
  external_id?: string | null;
  stdout_path?: string | null;
  stderr_path?: string | null;
}

export interface EnvironmentSessionSnapshot {
  session_id: string;
  component_id: string;
  config: Record<string, unknown>;
  step: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  done: boolean;
  action_count: number;
  action_labels: string[];
  observation_shape: number[];
  observation_dtype: string;
  observation_preview: unknown;
  observation_truncated: boolean;
  svg: string;
}

export interface DatasetArraySummary {
  name: string;
  shape: number[];
  dtype: string;
  min?: number | boolean | null;
  max?: number | boolean | null;
}

export interface DatasetListItem {
  path: string;
  size_bytes: number;
  modified_time: number;
}

export interface DatasetVisitation {
  height: number;
  width: number;
  action_count: number;
  action_labels: string[];
  valid_mask: boolean[][];
  state_counts: number[][];
  state_action_counts: number[][][];
  source: string;
}

export interface DatasetInspection {
  path: string;
  arrays: DatasetArraySummary[];
  is_transition_dataset: boolean;
  num_transitions?: number | null;
  preview: Array<Record<string, unknown>>;
  visitation?: DatasetVisitation | null;
}

export interface OfflineRndPoint {
  count: number;
  learned_bonus: number;
  count_bonus: number;
  row?: number | null;
  col?: number | null;
  action?: number | null;
}

export interface OfflineRndAnalysis {
  path: string;
  algorithm: "rnd" | "cfn" | "classifier" | "simhash" | string;
  granularity: "state" | "state_action" | string;
  epochs: number;
  batch_size: number;
  unique_items: number;
  loss_history: number[];
  visitation?: DatasetVisitation | null;
  learned_state_bonus?: Array<Array<number | null>> | null;
  count_state_bonus?: Array<Array<number | null>> | null;
  learned_state_action_bonus?: Array<Array<Array<number | null>>> | null;
  count_state_action_bonus?: Array<Array<Array<number | null>>> | null;
  scatter: OfflineRndPoint[];
}

export interface SweepCandidate {
  target: string;
  label: string;
  node_id: string;
  component: string;
  component_display_name: string;
  field: string;
  value: unknown;
  value_type: "integer" | "number" | "boolean" | "choice" | string;
  recommended_values: unknown[];
}

export interface SweepCandidateResponse {
  workflow_id: string;
  workflow_name: string;
  candidates: SweepCandidate[];
  seed_candidates: SweepCandidate[];
}

export interface SweepBuildParameter {
  label: string;
  target: string;
  values?: unknown[] | null;
  distribution?: "choice" | "uniform" | "loguniform" | "int_uniform";
  minimum?: number | null;
  maximum?: number | null;
}

export interface SweepBuildRequest {
  workflow_id: string;
  name?: string | null;
  description?: string;
  method: "grid" | "random";
  metric_name: string;
  metric_goal: "maximize" | "minimize";
  metric_last_n?: number | null;
  execution_backend?: "local" | "slurm" | null;
  parameters: SweepBuildParameter[];
  seed_target?: string | null;
  seed_start: number;
  seed_count: number;
  num_trials?: number | null;
  random_seed: number;
  slurm_max_parallel?: number | null;
  slurm_trials_per_task?: number;
  slurm_max_array_tasks?: number | null;
}

export interface SweepTrial {
  index: number;
  trial_id: string;
  group_id?: string | null;
  group_run_dir?: string | null;
  seed_value?: unknown | null;
  experiment_id: string;
  parameters: Record<string, unknown>;
  run_dir: string;
  command: string;
  workflow_path: string;
  metrics_path: string;
}

export interface SweepCompilation {
  sweep_id: string;
  name: string;
  method: "grid" | "random" | string;
  metric: { name: string; goal: "maximize" | "minimize" | string; last_n?: number | null };
  sweep_dir: string;
  manifest_path: string;
  slurm_array_path?: string | null;
  slurm_trials_per_task: number;
  slurm_array_task_count?: number | null;
  trials: SweepTrial[];
  generated_files: string[];
}

export interface SweepRunResponse {
  compilation: SweepCompilation;
  jobs: JobInfo[];
}

export interface SweepListItem {
  path: string;
  sweep_id: string;
  name: string;
  trial_count: number;
  modified_time: number;
}

export interface SweepInspectRequest {
  path: string;
  metric_name: string;
  metric_goal: "maximize" | "minimize";
  metric_last_n?: number | null;
}

export interface ExperimentHistoryPoint {
  episode: number;
  env_step?: number | null;
  return?: number | null;
  length?: number | null;
  loss?: number | null;
}

export interface ExperimentResult {
  experiment_id: string;
  status: string;
  run_dir: string;
  workflow_name: string;
  sweep_dir?: string | null;
  sweep_id?: string | null;
  sweep_trial_id?: string | null;
  sweep_group_id?: string | null;
  sweep_group_run_dir?: string | null;
  sweep_parameters: Record<string, unknown>;
  sweep_group_parameters?: Record<string, unknown>;
  seed?: unknown | null;
  metrics: Record<string, unknown>;
  train_history: ExperimentHistoryPoint[];
  eval_history: ExperimentHistoryPoint[];
}

export interface SweepTrialSummary {
  trial_id: string;
  experiment_id: string;
  run_dir: string;
  parameters: Record<string, unknown>;
  metric?: number | null;
}

export interface SweepGroupSummary {
  group_id: string;
  parameters: Record<string, unknown>;
  metric: number;
  metric_mean: number;
  metric_min: number;
  metric_max: number;
  metric_count: number;
  trial_ids: string[];
  run_dirs: string[];
}

export interface SweepSummary {
  sweep_id: string;
  metric: string;
  goal: "maximize" | "minimize" | string;
  metric_last_n?: number | null;
  best?: SweepGroupSummary | null;
  groups: SweepGroupSummary[];
  trials: SweepTrialSummary[];
}
