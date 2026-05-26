import type {
  ComponentSpec,
  DatasetListItem,
  DatasetInspection,
  OfflineRndAnalysis,
  EnvironmentSessionSnapshot,
  ExperimentSpec,
  JobInfo,
  ValidationResult,
  WorkflowGalleryItem,
  WorkflowSpec,
} from "../types/schema";

const defaultApiUrl =
  typeof window === "undefined"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname}:8000`;
const API_URL = import.meta.env.VITE_API_URL ?? defaultApiUrl;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const detail = await errorDetail(response);
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const detail = await errorDetail(response);
    throw new Error(detail || response.statusText);
  }
  return response.blob();
}

async function errorDetail(response: Response): Promise<string> {
  const detail = await response.text();
  if (!detail) return "";
  try {
    const parsed = JSON.parse(detail) as { detail?: unknown };
    return typeof parsed.detail === "string" ? parsed.detail : detail;
  } catch {
    return detail;
  }
}

export const api = {
  components: () => request<ComponentSpec[]>("/components"),
  workflowGallery: () => request<WorkflowGalleryItem[]>("/workflows"),
  saveWorkflow: (workflow: WorkflowSpec, workflowId?: string) =>
    request<WorkflowGalleryItem>("/workflows", {
      method: "POST",
      body: JSON.stringify({ workflow, workflow_id: workflowId }),
    }),
  loadWorkflow: (workflowId: string) => request<WorkflowSpec>(`/workflows/${encodeURIComponent(workflowId)}`),
  exampleWorkflow: (name: string) => request<WorkflowSpec>(`/workflows/examples/${name}`),
  validateWorkflow: (workflow: WorkflowSpec) =>
    request<ValidationResult>("/workflows/validate", { method: "POST", body: JSON.stringify(workflow) }),
  compileWorkflow: (workflow: WorkflowSpec) =>
    request<ExperimentSpec>("/workflows/compile", {
      method: "POST",
      body: JSON.stringify({ workflow }),
    }),
  runWorkflow: (workflow: WorkflowSpec, backend: "local" | "slurm") =>
    request<JobInfo>("/experiments/run", {
      method: "POST",
      body: JSON.stringify({ workflow, backend }),
    }),
  jobs: () => request<JobInfo[]>("/jobs"),
  jobLogs: (jobId: string) => request<string>(`/jobs/${jobId}/logs`),
  experiments: () => request<Array<Record<string, unknown>>>("/experiments"),
  artifacts: (experimentId: string) => request<string[]>(`/artifacts/${experimentId}`),
  createEnvironmentSession: (payload: { component_id: string; config: Record<string, unknown>; seed: number }) =>
    request<EnvironmentSessionSnapshot>("/environment-sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  stepEnvironmentSession: (sessionId: string, action: number) =>
    request<EnvironmentSessionSnapshot>(`/environment-sessions/${sessionId}/actions`, {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  resetEnvironmentSession: (sessionId: string) =>
    request<EnvironmentSessionSnapshot>(`/environment-sessions/${sessionId}/reset`, { method: "POST" }),
  exportEnvironmentPdf: (sessionId: string) =>
    requestBlob(`/environment-sessions/${sessionId}/export.pdf`, { method: "GET" }),
  datasets: () => request<DatasetListItem[]>("/datasets"),
  inspectDataset: (path: string, previewRows = 25) =>
    request<DatasetInspection>("/datasets/inspect", {
      method: "POST",
      body: JSON.stringify({ path, preview_rows: previewRows }),
    }),
  trainOfflineRnd: (payload: {
    path: string;
    algorithm: "rnd" | "cfn" | "classifier";
    granularity: "state" | "state_action";
    epochs: number;
    batch_size: number;
    learning_rate: number;
    hidden_units: number[];
    activation: "relu" | "tanh" | "gelu" | "elu" | "linear";
    optimizer: "adam" | "sgd" | "rmsprop";
    action_conditioning: "none" | "input" | "output" | "pair";
    update_period: number;
    output_dim: number;
    intrinsic_reward_scale: number;
    intrinsic_stats_decay: number;
    intrinsic_reward_epsilon: number;
    intrinsic_reward_clip: number | null;
    intrinsic_reward_center: boolean;
    max_grad_norm: number;
    seed: number;
  }) =>
    request<OfflineRndAnalysis>("/datasets/offline-rnd", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
