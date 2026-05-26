import { useMutation } from "@tanstack/react-query";
import { Download, Play, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { ComponentSpec, EnvironmentSessionSnapshot } from "../../types/schema";
import { SchemaForm } from "../forms/SchemaForm";

interface EnvironmentPlaygroundProps {
  components: ComponentSpec[];
  isLoading?: boolean;
}

export function EnvironmentPlayground({ components, isLoading }: EnvironmentPlaygroundProps) {
  const renderableEnvironments = useMemo(
    () => components.filter((component) => component.kind === "environment" && component.id === "navix.env.grid"),
    [components],
  );
  const [componentId, setComponentId] = useState("navix.env.grid");
  const selected = renderableEnvironments.find((component) => component.id === componentId) ?? renderableEnvironments[0];
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [seed, setSeed] = useState(0);
  const [snapshot, setSnapshot] = useState<EnvironmentSessionSnapshot>();

  const resolvedConfig = useMemo(() => ({ ...(selected?.defaults ?? {}), ...config }), [selected, config]);

  const load = useMutation({
    mutationFn: () => api.createEnvironmentSession({ component_id: selected?.id ?? componentId, config: resolvedConfig, seed }),
    onSuccess: setSnapshot,
  });
  const step = useMutation({
    mutationFn: (action: number) => {
      if (!snapshot) throw new Error("No environment session loaded");
      return api.stepEnvironmentSession(snapshot.session_id, action);
    },
    onSuccess: setSnapshot,
  });
  const reset = useMutation({
    mutationFn: () => {
      if (!snapshot) throw new Error("No environment session loaded");
      return api.resetEnvironmentSession(snapshot.session_id);
    },
    onSuccess: setSnapshot,
  });
  const exportPdf = useMutation({
    mutationFn: async () => {
      if (!snapshot) throw new Error("No environment session loaded");
      const blob = await api.exportEnvironmentPdf(snapshot.session_id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${snapshot.component_id.replace(/\./g, "_")}_step_${snapshot.step}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);
    },
  });

  const isBusy = load.isPending || step.isPending || reset.isPending || exportPdf.isPending;
  const error = load.error ?? step.error ?? reset.error ?? exportPdf.error;

  return (
    <main className="environment-page">
      <section className="environment-controls">
        <div className="page-header compact">
          <h1>Environment</h1>
          <button onClick={() => load.mutate()} disabled={!selected || isBusy}>
            <Play size={16} />
            Load
          </button>
        </div>
        {isLoading && <div className="empty-state">Loading environments</div>}
        {!isLoading && renderableEnvironments.length === 0 && (
          <div className="empty-state">No renderable environment components registered</div>
        )}
        {selected && (
          <>
            <label className="field">
              <span>component</span>
              <select
                value={selected.id}
                onChange={(event) => {
                  setComponentId(event.target.value);
                  setConfig({});
                  setSnapshot(undefined);
                }}
              >
                {renderableEnvironments.map((component) => (
                  <option key={component.id} value={component.id}>
                    {component.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>seed</span>
              <input
                type="number"
                min={0}
                step={1}
                value={seed}
                onChange={(event) => setSeed(Number.parseInt(event.target.value, 10) || 0)}
              />
            </label>
            <SchemaForm schema={selected.config_schema} value={resolvedConfig} onChange={setConfig} />
          </>
        )}
        {snapshot && (
          <>
            <div className="environment-status">
              <div>
                <span>step</span>
                <strong>{snapshot.step}</strong>
              </div>
              <div>
                <span>reward</span>
                <strong>{snapshot.reward.toFixed(3)}</strong>
              </div>
              <div>
                <span>done</span>
                <strong>{snapshot.done ? "yes" : "no"}</strong>
              </div>
            </div>
            <div className="observation-panel">
              <div className="panel-title">Observation</div>
              <dl>
                <div>
                  <dt>shape</dt>
                  <dd>{snapshot.observation_shape.length ? `[${snapshot.observation_shape.join(", ")}]` : "scalar"}</dd>
                </div>
                <div>
                  <dt>dtype</dt>
                  <dd>{snapshot.observation_dtype}</dd>
                </div>
              </dl>
              <pre>{JSON.stringify(snapshot.observation_preview, null, 2)}</pre>
              {snapshot.observation_truncated && <small>Preview truncated</small>}
            </div>
          </>
        )}
        {error && <div className="error-state">{error.message}</div>}
      </section>
      <section className="environment-stage">
        <div className="environment-toolbar">
          <div className="action-row">
            {snapshot?.action_labels.map((label, action) => (
              <button key={`${label}-${action}`} onClick={() => step.mutate(action)} disabled={isBusy}>
                {label}
              </button>
            ))}
          </div>
          <div className="action-row">
            <button onClick={() => reset.mutate()} disabled={!snapshot || isBusy}>
              <RotateCcw size={16} />
              Reset
            </button>
            <button onClick={() => exportPdf.mutate()} disabled={!snapshot || isBusy}>
              <Download size={16} />
              PDF
            </button>
          </div>
        </div>
        <div className="environment-render">
          {snapshot ? (
            <div dangerouslySetInnerHTML={{ __html: snapshot.svg }} />
          ) : (
            <div className="empty-state">Load an environment to preview and step through it</div>
          )}
        </div>
      </section>
    </main>
  );
}
