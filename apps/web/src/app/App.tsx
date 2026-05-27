import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Boxes, Brain, Database, FlaskConical, FolderOpen, GitBranch, ListChecks, Map, RefreshCw, Save, SlidersHorizontal } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import { ComponentPalette } from "../components/flow/ComponentPalette";
import { EnvironmentPlayground } from "../components/environments/EnvironmentPlayground";
import { DatasetPage } from "../components/datasets/DatasetPage";
import { OfflineRlPage } from "../components/datasets/OfflineRlPage";
import { SweepPage } from "../components/sweeps/SweepPage";
import { InspectorPanel } from "../components/flow/InspectorPanel";
import { ValidationPanel } from "../components/flow/ValidationPanel";
import { WorkflowCanvas } from "../components/flow/WorkflowCanvas";
import { ExperimentDetail } from "../components/jobs/ExperimentDetail";
import { JobsPage } from "../components/jobs/JobsPage";
import { useFlowStore } from "../stores/flowStore";
import type { ExperimentSpec, ValidationResult } from "../types/schema";

type Page = "flow" | "jobs" | "experiment" | "environment" | "dataset" | "offline" | "sweep";

export function App() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState<Page>("flow");
  const [validation, setValidation] = useState<ValidationResult>();
  const [experiment, setExperiment] = useState<ExperimentSpec>();
  const [selectedWorkflowId, setSelectedWorkflowId] = useState("");
  const components = useQuery({ queryKey: ["components"], queryFn: api.components });
  const workflowGallery = useQuery({ queryKey: ["workflow-gallery"], queryFn: api.workflowGallery });
  const workflowName = useFlowStore((state) => state.workflowName);
  const setWorkflowName = useFlowStore((state) => state.setWorkflowName);
  const toWorkflow = useFlowStore((state) => state.toWorkflow);
  const loadWorkflow = useFlowStore((state) => state.loadWorkflow);

  const validate = useMutation({
    mutationFn: () => api.validateWorkflow(toWorkflow()),
    onSuccess: setValidation,
  });
  const compile = useMutation({
    mutationFn: () => api.compileWorkflow(toWorkflow()),
    onSuccess: setExperiment,
  });
  const run = useMutation({
    mutationFn: (backend: "local" | "slurm") => api.runWorkflow(toWorkflow(), backend),
  });
  const saveWorkflow = useMutation({
    mutationFn: () => api.saveWorkflow(toWorkflow()),
    onSuccess: (item) => {
      setSelectedWorkflowId(item.workflow_id);
      void queryClient.invalidateQueries({ queryKey: ["workflow-gallery"] });
    },
  });
  const loadSavedWorkflow = useMutation({
    mutationFn: (workflowId: string) => api.loadWorkflow(workflowId),
    onSuccess: (workflow) => {
      if (!components.data) return;
      loadWorkflow(workflow, components.data);
      setValidation(undefined);
      setExperiment(undefined);
    },
  });

  return (
    <div className="app-shell">
      <nav className="topbar">
        <div className="brand">
          <Boxes size={20} />
          rl-flow
        </div>
        <button className={page === "flow" ? "active" : ""} onClick={() => setPage("flow")}>
          <GitBranch size={16} />
          Flow
        </button>
        <button className={page === "jobs" ? "active" : ""} onClick={() => setPage("jobs")}>
          <ListChecks size={16} />
          Jobs
        </button>
        <button className={page === "environment" ? "active" : ""} onClick={() => setPage("environment")}>
          <Map size={16} />
          Environment
        </button>
        <button className={page === "dataset" ? "active" : ""} onClick={() => setPage("dataset")}>
          <Database size={16} />
          Dataset
        </button>
        <button className={page === "offline" ? "active" : ""} onClick={() => setPage("offline")}>
          <Brain size={16} />
          Offline RL
        </button>
        <button className={page === "sweep" ? "active" : ""} onClick={() => setPage("sweep")}>
          <SlidersHorizontal size={16} />
          Sweep
        </button>
        <button className={page === "experiment" ? "active" : ""} onClick={() => setPage("experiment")}>
          <FlaskConical size={16} />
          Experiment
        </button>
        <div className="workflow-controls">
          <input
            className="workflow-name-input"
            aria-label="Workflow name"
            value={workflowName}
            onChange={(event) => setWorkflowName(event.target.value)}
          />
          <button
            title="Save workflow"
            onClick={() => saveWorkflow.mutate()}
            disabled={saveWorkflow.isPending || workflowName.trim().length === 0}
          >
            <Save size={16} />
            Save
          </button>
          <select
            className="gallery-select"
            aria-label="Saved workflows"
            value={selectedWorkflowId}
            onChange={(event) => setSelectedWorkflowId(event.target.value)}
          >
            <option value="">Gallery</option>
            {(workflowGallery.data ?? []).map((item) => (
              <option key={item.workflow_id} value={item.workflow_id}>
                {item.name}
              </option>
            ))}
          </select>
          <button
            title="Load saved workflow"
            onClick={() => loadSavedWorkflow.mutate(selectedWorkflowId)}
            disabled={!components.data || !selectedWorkflowId || loadSavedWorkflow.isPending}
          >
            <FolderOpen size={16} />
            Load
          </button>
          <button title="Refresh gallery" onClick={() => workflowGallery.refetch()} disabled={workflowGallery.isFetching}>
            <RefreshCw size={16} />
          </button>
        </div>
      </nav>
      {page === "flow" && (
        <div className="flow-layout">
          <ComponentPalette
            components={components.data ?? []}
            isLoading={components.isLoading}
            error={components.error}
          />
          <WorkflowCanvas components={components.data ?? []} />
          <InspectorPanel />
          <ValidationPanel
            validation={validation}
            experiment={experiment}
            isBusy={validate.isPending || compile.isPending || run.isPending}
            onValidate={() => validate.mutate()}
            onCompile={() => compile.mutate()}
            onRun={(backend) => run.mutate(backend)}
          />
        </div>
      )}
      {page === "jobs" && <JobsPage />}
      {page === "environment" && <EnvironmentPlayground components={components.data ?? []} isLoading={components.isLoading} />}
      {page === "dataset" && <DatasetPage />}
      {page === "offline" && <OfflineRlPage />}
      {page === "sweep" && <SweepPage />}
      {page === "experiment" && <ExperimentDetail />}
    </div>
  );
}
