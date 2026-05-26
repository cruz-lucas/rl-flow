import { useMutation, useQuery } from "@tanstack/react-query";
import { Database, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ActionVisitationHeatmap, StateVisitationHeatmap } from "./Heatmaps";

type DatasetTab = "preview" | "visitation";
type VisitationMode = "state" | "state_action";

export function DatasetPage() {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [path, setPath] = useState("");
  const [previewRows, setPreviewRows] = useState(25);
  const [tab, setTab] = useState<DatasetTab>("preview");
  const [visitationMode, setVisitationMode] = useState<VisitationMode>("state");
  const inspect = useMutation({
    mutationFn: () => api.inspectDataset(path, previewRows),
  });

  useEffect(() => {
    if (!path && datasets.data?.[0]) {
      setPath(datasets.data[0].path);
    }
  }, [datasets.data, path]);

  const dataset = inspect.data;
  return (
    <main className="page dataset-page">
      <div className="page-header">
        <h1>
          <Database size={20} />
          Dataset
        </h1>
        <button onClick={() => inspect.mutate()} disabled={inspect.isPending || path.trim().length === 0}>
          <Search size={16} />
          Inspect
        </button>
      </div>
      <div className="dataset-controls">
        <label className="field">
          <span>dataset path</span>
          <input value={path} onChange={(event) => setPath(event.target.value)} />
        </label>
        <label className="field">
          <span>recent dataset</span>
          <select value={path} onChange={(event) => setPath(event.target.value)}>
            <option value="">Select</option>
            {(datasets.data ?? []).map((item) => (
              <option key={item.path} value={item.path}>
                {item.path}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>preview rows</span>
          <input
            type="number"
            min={1}
            max={500}
            step={1}
            value={previewRows}
            onChange={(event) => setPreviewRows(Number.parseInt(event.target.value, 10) || 1)}
          />
        </label>
      </div>
      {inspect.error && <div className="error-state">{inspect.error.message}</div>}
      {dataset && (
        <>
          <div className="segmented-control">
            <button className={tab === "preview" ? "active" : ""} onClick={() => setTab("preview")}>
              Preview
            </button>
            <button className={tab === "visitation" ? "active" : ""} onClick={() => setTab("visitation")}>
              Visitation
            </button>
          </div>
          {tab === "preview" && (
            <div className="dataset-grid">
              <section>
                <div className="panel-title">Arrays</div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Shape</th>
                      <th>Dtype</th>
                      <th>Min</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.arrays.map((array) => (
                      <tr key={array.name}>
                        <td>{array.name}</td>
                        <td>{array.shape.length ? `[${array.shape.join(", ")}]` : "scalar"}</td>
                        <td>{array.dtype}</td>
                        <td>{String(array.min ?? "")}</td>
                        <td>{String(array.max ?? "")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
              <section>
                <div className="panel-title">
                  Transitions
                  {dataset.num_transitions !== null && dataset.num_transitions !== undefined && (
                    <span className="source-count">{dataset.num_transitions}</span>
                  )}
                </div>
                {dataset.is_transition_dataset ? (
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>s</th>
                          <th>a</th>
                          <th>r</th>
                          <th>s'</th>
                          <th>done</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dataset.preview.map((row) => (
                          <tr key={String(row.index)}>
                            <td>{String(row.index)}</td>
                            <td>{formatValue(row.observation)}</td>
                            <td>{formatValue(row.action)}</td>
                            <td>{formatValue(row.reward)}</td>
                            <td>{formatValue(row.next_observation)}</td>
                            <td>{formatValue(row.terminal)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty-state">This file is not a recognized transition dataset</div>
                )}
              </section>
            </div>
          )}
          {tab === "visitation" && (
            <div className="dataset-grid">
              {dataset.visitation ? (
                <>
                  <div className="segmented-control compact">
                    <button
                      className={visitationMode === "state" ? "active" : ""}
                      onClick={() => setVisitationMode("state")}
                    >
                      State
                    </button>
                    <button
                      className={visitationMode === "state_action" ? "active" : ""}
                      onClick={() => setVisitationMode("state_action")}
                    >
                      State-Action
                    </button>
                  </div>
                  {visitationMode === "state" ? (
                    <StateVisitationHeatmap visitation={dataset.visitation} />
                  ) : (
                    <ActionVisitationHeatmap visitation={dataset.visitation} />
                  )}
                </>
              ) : (
                <div className="empty-state">No grid visitation map is available for this dataset</div>
              )}
            </div>
          )}
        </>
      )}
    </main>
  );
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  return String(value);
}
