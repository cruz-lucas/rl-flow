import { useMutation, useQuery } from "@tanstack/react-query";
import { Database, Eye, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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
  const [selectedPreviewIndex, setSelectedPreviewIndex] = useState(0);
  const inspect = useMutation({
    mutationFn: () => api.inspectDataset(path, previewRows),
  });

  useEffect(() => {
    if (!path && datasets.data?.[0]) {
      setPath(datasets.data[0].path);
    }
  }, [datasets.data, path]);

  const dataset = inspect.data;
  const selectedPreview = dataset?.preview[selectedPreviewIndex] ?? dataset?.preview[0];
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
                          <th />
                          <th>s</th>
                          <th>a</th>
                          <th>r</th>
                          <th>s'</th>
                          <th>done</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dataset.preview.map((row, rowIndex) => (
                          <tr key={String(row.index)}>
                            <td>{String(row.index)}</td>
                            <td>
                              <button className="icon-button" onClick={() => setSelectedPreviewIndex(rowIndex)}>
                                <Eye size={16} />
                              </button>
                            </td>
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
              {dataset.is_transition_dataset && selectedPreview && (
                <section>
                  <div className="panel-title">Observation</div>
                  <div className="observation-preview-grid">
                    <SymbolicObservationPanel title="s" value={selectedPreview.observation} />
                    <SymbolicObservationPanel title="s'" value={selectedPreview.next_observation} />
                  </div>
                </section>
              )}
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
  const symbolic = decodeSymbolicObservation(value);
  if (symbolic) {
    return `symbolic ${symbolic.width}x${symbolic.height}x3, wall colors ${symbolic.wallColors.join(", ") || "none"}`;
  }
  if (Array.isArray(value)) {
    const text = JSON.stringify(value);
    return text.length > 180 ? `${text.slice(0, 180)}...` : text;
  }
  return String(value);
}

function SymbolicObservationPanel({ title, value }: { title: string; value: unknown }) {
  const symbolic = useMemo(() => decodeSymbolicObservation(value), [value]);
  if (!symbolic) {
    return (
      <div className="symbolic-preview-panel">
        <div className="panel-title">{title}</div>
        <pre>{formatValue(value)}</pre>
      </div>
    );
  }
  const cell = 18;
  return (
    <div className="symbolic-preview-panel">
      <div className="plot-header">
        <div className="panel-title">{title}</div>
        <span className="source-count">{symbolic.wallColors.length ? `walls ${symbolic.wallColors.join(", ")}` : ""}</span>
      </div>
      <svg viewBox={`0 0 ${symbolic.width * cell} ${symbolic.height * cell}`} className="symbolic-preview-svg" role="img">
        {symbolic.grid.map((row, rowIndex) =>
          row.map((cellValue, colIndex) => (
            <g key={`${rowIndex}-${colIndex}`}>
              <rect
                x={colIndex * cell}
                y={rowIndex * cell}
                width={cell}
                height={cell}
                fill={cellFill(cellValue)}
              />
              {cellValue[0] === 10 && (
                <circle
                  cx={colIndex * cell + cell / 2}
                  cy={rowIndex * cell + cell / 2}
                  r={cell * 0.28}
                  fill="#1f5f6f"
                />
              )}
              {cellValue[0] === 8 && (
                <circle
                  cx={colIndex * cell + cell / 2}
                  cy={rowIndex * cell + cell / 2}
                  r={cell * 0.25}
                  fill="#44a366"
                />
              )}
            </g>
          )),
        )}
      </svg>
    </div>
  );
}

function decodeSymbolicObservation(value: unknown):
  | { grid: number[][][]; width: number; height: number; wallColors: number[] }
  | null {
  const raw = flattenNumbers(value);
  if (!raw.length || raw.length % 3 !== 0) return null;
  const side = Math.round(Math.sqrt(raw.length / 3));
  if (side * side * 3 !== raw.length) return null;
  const scaled = Math.max(...raw) <= 1 ? raw.map((item) => Math.round(item * 255)) : raw.map((item) => Math.round(item));
  const grid: number[][][] = [];
  let hasPlayer = false;
  const wallColors = new Set<number>();
  for (let row = 0; row < side; row += 1) {
    const gridRow: number[][] = [];
    for (let col = 0; col < side; col += 1) {
      const offset = (row * side + col) * 3;
      const cell = [scaled[offset], scaled[offset + 1], scaled[offset + 2]];
      if (cell[0] === 10) hasPlayer = true;
      if (cell[0] === 2) wallColors.add(cell[1]);
      gridRow.push(cell);
    }
    grid.push(gridRow);
  }
  if (!hasPlayer && !wallColors.size) return null;
  return { grid, width: side, height: side, wallColors: [...wallColors].sort((a, b) => a - b).slice(0, 12) };
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (!Array.isArray(value)) return [];
  const output: number[] = [];
  for (const item of value) {
    output.push(...flattenNumbers(item));
  }
  return output;
}

function cellFill(cell: number[]): string {
  if (cell[0] === 2) {
    return cell[1] === 5 ? "#2f3a45" : wallColor(cell[1]);
  }
  return "#fbfcfd";
}

function wallColor(colour: number): string {
  const hue = (colour * 137) % 360;
  const saturation = 48 + (colour % 32);
  const lightness = 30 + (Math.floor(colour / 8) % 26);
  return hslToHex(hue / 360, saturation / 100, lightness / 100);
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  function channel(offset: number) {
    const k = (offset + hue * 12) % 12;
    const a = saturation * Math.min(lightness, 1 - lightness);
    const value = lightness - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(255 * value)
      .toString(16)
      .padStart(2, "0");
  }
  return `#${channel(0)}${channel(8)}${channel(4)}`;
}
