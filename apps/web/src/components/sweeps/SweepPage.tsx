import { useMutation, useQuery } from "@tanstack/react-query";
import { Play, Plus, SlidersHorizontal, Wand2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { SweepBuildRequest, SweepCandidate, SweepCompilation, SweepRunResponse } from "../../types/schema";

type SelectedParameter = {
  target: string;
  label: string;
  valueType: string;
  valuesText: string;
};

export function SweepPage() {
  const workflows = useQuery({ queryKey: ["workflow-gallery"], queryFn: api.workflowGallery });
  const [workflowId, setWorkflowId] = useState("");
  const [name, setName] = useState("");
  const [method, setMethod] = useState<"grid" | "random">("grid");
  const [metricName, setMetricName] = useState("mean_eval_return");
  const [metricGoal, setMetricGoal] = useState<"maximize" | "minimize">("maximize");
  const [metricLastN, setMetricLastN] = useState(50);
  const [backend, setBackend] = useState<"inherit" | "local" | "slurm">("inherit");
  const [seedTarget, setSeedTarget] = useState("");
  const [seedStart, setSeedStart] = useState(0);
  const [seedCount, setSeedCount] = useState(3);
  const [numTrials, setNumTrials] = useState(20);
  const [randomSeed, setRandomSeed] = useState(0);
  const [slurmMaxParallel, setSlurmMaxParallel] = useState(8);
  const [selected, setSelected] = useState<SelectedParameter[]>([]);

  const candidates = useQuery({
    queryKey: ["sweep-candidates", workflowId],
    queryFn: () => api.sweepCandidates(workflowId),
    enabled: workflowId.length > 0,
  });
  const compile = useMutation({ mutationFn: () => api.compileSweep(buildPayload()) });
  const run = useMutation({ mutationFn: () => api.runSweep(buildPayload()) });

  useEffect(() => {
    if (!workflowId && workflows.data?.[0]) {
      setWorkflowId(workflows.data[0].workflow_id);
    }
  }, [workflowId, workflows.data]);

  useEffect(() => {
    const workflow = workflows.data?.find((item) => item.workflow_id === workflowId);
    setName(workflow ? `${workflow.name} sweep` : "");
    setSelected([]);
  }, [workflowId, workflows.data]);

  useEffect(() => {
    if (!seedTarget && candidates.data?.seed_candidates[0]) {
      setSeedTarget(candidates.data.seed_candidates[0].target);
    }
  }, [candidates.data, seedTarget]);

  const selectedTargets = useMemo(() => new Set(selected.map((item) => item.target)), [selected]);
  const result = run.data?.compilation ?? compile.data;

  function buildPayload(): SweepBuildRequest {
    return {
      workflow_id: workflowId,
      name,
      method,
      metric_name: metricName,
      metric_goal: metricGoal,
      metric_last_n: metricName === "mean_train_return_last_n" ? metricLastN : null,
      execution_backend: backend === "inherit" ? null : backend,
      parameters: selected.map((parameter) => ({
        label: parameter.label,
        target: parameter.target,
        values: parseValues(parameter.valuesText, parameter.valueType),
      })),
      seed_target: seedTarget || null,
      seed_start: seedStart,
      seed_count: seedCount,
      num_trials: method === "random" ? numTrials : null,
      random_seed: randomSeed,
      slurm_max_parallel: backend === "slurm" ? slurmMaxParallel : null,
    };
  }

  function addCandidate(candidate: SweepCandidate) {
    if (selectedTargets.has(candidate.target)) return;
    setSelected((items) => [
      ...items,
      {
        target: candidate.target,
        label: candidate.label.replace(/\./g, "_"),
        valueType: candidate.value_type,
        valuesText: valuesText(candidate.recommended_values.length ? candidate.recommended_values : [candidate.value]),
      },
    ]);
  }

  function updateSelected(target: string, patch: Partial<SelectedParameter>) {
    setSelected((items) => items.map((item) => (item.target === target ? { ...item, ...patch } : item)));
  }

  return (
    <main className="page sweep-page">
      <div className="page-header">
        <h1>
          <SlidersHorizontal size={20} />
          Sweep
        </h1>
        <div className="action-row">
          <button onClick={() => compile.mutate()} disabled={!workflowId || compile.isPending || selected.length + seedCount === 0}>
            <Wand2 size={16} />
            Compile
          </button>
          <button onClick={() => run.mutate()} disabled={!workflowId || run.isPending || selected.length + seedCount === 0}>
            <Play size={16} />
            Run
          </button>
        </div>
      </div>

      <div className="sweep-layout">
        <section className="sweep-panel">
          <div className="panel-title">Setup</div>
          <div className="sweep-controls">
            <label className="field wide">
              <span>workflow</span>
              <select value={workflowId} onChange={(event) => setWorkflowId(event.target.value)}>
                <option value="">Select</option>
                {(workflows.data ?? []).map((item) => (
                  <option key={item.workflow_id} value={item.workflow_id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field wide">
              <span>sweep name</span>
              <input value={name} onChange={(event) => setName(event.target.value)} />
            </label>
            <label className="field">
              <span>method</span>
              <select value={method} onChange={(event) => setMethod(event.target.value as "grid" | "random")}>
                <option value="grid">grid</option>
                <option value="random">random</option>
              </select>
            </label>
            <label className="field">
              <span>backend</span>
              <select value={backend} onChange={(event) => setBackend(event.target.value as typeof backend)}>
                <option value="inherit">inherit</option>
                <option value="local">local</option>
                <option value="slurm">slurm</option>
              </select>
            </label>
            <label className="field">
              <span>metric</span>
              <select value={metricName} onChange={(event) => setMetricName(event.target.value)}>
                <option value="mean_eval_return">mean eval return</option>
                <option value="mean_train_return">mean train return</option>
                <option value="mean_train_return_last_n">mean train return, last N</option>
              </select>
            </label>
            <label className="field">
              <span>goal</span>
              <select value={metricGoal} onChange={(event) => setMetricGoal(event.target.value as "maximize" | "minimize")}>
                <option value="maximize">maximize</option>
                <option value="minimize">minimize</option>
              </select>
            </label>
            {metricName === "mean_train_return_last_n" && (
              <label className="field">
                <span>last N</span>
                <input type="number" min={1} step={1} value={metricLastN} onChange={(event) => setMetricLastN(intValue(event.target.value, 1))} />
              </label>
            )}
            <label className="field">
              <span>seed target</span>
              <select value={seedTarget} onChange={(event) => setSeedTarget(event.target.value)}>
                <option value="">none</option>
                {(candidates.data?.seed_candidates ?? []).map((candidate) => (
                  <option key={candidate.target} value={candidate.target}>
                    {candidate.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>seed count</span>
              <input type="number" min={0} step={1} value={seedCount} onChange={(event) => setSeedCount(intValue(event.target.value, 0))} />
            </label>
            <label className="field">
              <span>seed start</span>
              <input type="number" min={0} step={1} value={seedStart} onChange={(event) => setSeedStart(intValue(event.target.value, 0))} />
            </label>
            {method === "random" && (
              <>
                <label className="field">
                  <span>trials</span>
                  <input type="number" min={1} step={1} value={numTrials} onChange={(event) => setNumTrials(intValue(event.target.value, 1))} />
                </label>
                <label className="field">
                  <span>random seed</span>
                  <input type="number" min={0} step={1} value={randomSeed} onChange={(event) => setRandomSeed(intValue(event.target.value, 0))} />
                </label>
              </>
            )}
            {backend === "slurm" && (
              <label className="field">
                <span>max parallel</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={slurmMaxParallel}
                  onChange={(event) => setSlurmMaxParallel(intValue(event.target.value, 1))}
                />
              </label>
            )}
          </div>
          {(compile.error || run.error || candidates.error) && (
            <div className="error-state">{(compile.error ?? run.error ?? candidates.error)?.message}</div>
          )}
        </section>

        <section className="sweep-panel">
          <div className="panel-title">Parameters</div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Value</th>
                  <th>Values</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {(candidates.data?.candidates ?? []).map((candidate) => (
                  <tr key={candidate.target}>
                    <td>
                      <strong>{candidate.label}</strong>
                      <span className="muted-line">{candidate.component_display_name}</span>
                    </td>
                    <td>{formatCandidateValue(candidate.value)}</td>
                    <td>{valuesText(candidate.recommended_values)}</td>
                    <td>
                      <button className="icon-button" onClick={() => addCandidate(candidate)} disabled={selectedTargets.has(candidate.target)}>
                        <Plus size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="sweep-panel">
          <div className="panel-title">Selected</div>
          {selected.length ? (
            <div className="selected-sweep-params">
              {selected.map((parameter) => (
                <div className="sweep-param-row" key={parameter.target}>
                  <label className="field">
                    <span>label</span>
                    <input value={parameter.label} onChange={(event) => updateSelected(parameter.target, { label: event.target.value })} />
                  </label>
                  <label className="field wide">
                    <span>{parameter.target}</span>
                    <input value={parameter.valuesText} onChange={(event) => updateSelected(parameter.target, { valuesText: event.target.value })} />
                  </label>
                  <button className="icon-button" onClick={() => setSelected((items) => items.filter((item) => item.target !== parameter.target))}>
                    <X size={16} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">No parameters selected</div>
          )}
        </section>

        <SweepResult compilation={result} run={run.data} />
      </div>
    </main>
  );
}

function SweepResult({ compilation, run }: { compilation?: SweepCompilation; run?: SweepRunResponse }) {
  if (!compilation) return null;
  return (
    <section className="sweep-panel sweep-result">
      <div className="panel-title">Result</div>
      <div className="summary-strip">
        <div>
          <span>trials</span>
          <strong>{compilation.trials.length}</strong>
        </div>
        <div>
          <span>method</span>
          <strong>{compilation.method}</strong>
        </div>
        <div>
          <span>jobs</span>
          <strong>{run?.jobs.length ?? 0}</strong>
        </div>
      </div>
      <dl className="sweep-paths">
        <dt>sweep dir</dt>
        <dd>{compilation.sweep_dir}</dd>
        <dt>manifest</dt>
        <dd>{compilation.manifest_path}</dd>
        {compilation.slurm_array_path && (
          <>
            <dt>slurm</dt>
            <dd>{compilation.slurm_array_path}</dd>
          </>
        )}
      </dl>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Trial</th>
              <th>Experiment</th>
              <th>Parameters</th>
            </tr>
          </thead>
          <tbody>
            {compilation.trials.slice(0, 100).map((trial) => (
              <tr key={trial.trial_id}>
                <td>{trial.trial_id}</td>
                <td>{trial.experiment_id}</td>
                <td>{JSON.stringify(trial.parameters)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function parseValues(text: string, valueType: string): unknown[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed) as unknown;
    return Array.isArray(parsed) ? parsed : [parsed];
  }
  return trimmed
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => coerceValue(item, valueType));
}

function coerceValue(value: string, valueType: string): unknown {
  if (valueType === "integer") return Number.parseInt(value, 10);
  if (valueType === "number") return Number.parseFloat(value);
  if (valueType === "boolean") return value.toLowerCase() === "true";
  return value;
}

function valuesText(values: unknown[]): string {
  return values.map((value) => (typeof value === "string" ? value : JSON.stringify(value) ?? String(value))).join(", ");
}

function intValue(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatCandidateValue(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value) ?? String(value);
}
