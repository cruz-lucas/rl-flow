import { useQuery } from "@tanstack/react-query";
import { BarChart3, Download, FlaskConical, LineChart } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { ExperimentHistoryPoint, ExperimentResult, SweepGroupSummary, SweepSummary } from "../../types/schema";

type MetricGoal = "maximize" | "minimize";

export function ExperimentDetail() {
  const results = useQuery({ queryKey: ["experiment-results"], queryFn: api.experimentResults });
  const sweeps = useQuery({ queryKey: ["sweeps"], queryFn: api.sweeps });
  const [experimentId, setExperimentId] = useState("");
  const [sweepPath, setSweepPath] = useState("");
  const [metricName, setMetricName] = useState("mean_eval_return");
  const [metricGoal, setMetricGoal] = useState<MetricGoal>("maximize");
  const [metricLastN, setMetricLastN] = useState(50);

  useEffect(() => {
    if (!experimentId && results.data?.[0]) {
      setExperimentId(results.data[0].experiment_id);
    }
  }, [experimentId, results.data]);

  useEffect(() => {
    if (!sweepPath && sweeps.data?.[0]) {
      setSweepPath(sweeps.data[0].path);
    }
  }, [sweepPath, sweeps.data]);

  const selectedRun = useMemo(
    () => results.data?.find((item) => item.experiment_id === experimentId) ?? results.data?.[0],
    [experimentId, results.data],
  );
  const selectedRunGroup = useMemo(() => {
    if (!selectedRun) return [];
    if (!selectedRun.sweep_id || !selectedRun.sweep_group_id) return [selectedRun];
    return (results.data ?? []).filter(
      (item) => item.sweep_id === selectedRun.sweep_id && item.sweep_group_id === selectedRun.sweep_group_id,
    );
  }, [results.data, selectedRun]);
  const sweepSummary = useQuery({
    queryKey: ["sweep-summary", sweepPath, metricName, metricGoal, metricLastN],
    queryFn: () =>
      api.inspectSweep({
        path: sweepPath,
        metric_name: metricName,
        metric_goal: metricGoal,
        metric_last_n: metricName === "mean_train_return_last_n" ? metricLastN : null,
      }),
    enabled: Boolean(sweepPath),
  });

  return (
    <main className="page experiment-results-page">
      <header className="page-header">
        <h1>
          <FlaskConical size={20} />
          Experiment Results
        </h1>
      </header>

      <div className="results-layout">
        <section className="results-panel">
          <div className="plot-header">
            <div className="panel-title">Run Results</div>
            <label className="field compact-field">
              <span>run</span>
              <select value={selectedRun?.experiment_id ?? ""} onChange={(event) => setExperimentId(event.target.value)}>
                <option value="">Select</option>
                {(results.data ?? []).map((item) => (
                  <option key={item.experiment_id} value={item.experiment_id}>
                    {item.experiment_id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {results.error && <div className="error-state">{results.error.message}</div>}
          {!selectedRun ? (
            <div className="empty-state">No run results</div>
          ) : (
            <RunResults run={selectedRun} groupRuns={selectedRunGroup} />
          )}
        </section>

        <section className="results-panel">
          <div className="plot-header">
            <div className="panel-title">Sweep Results</div>
          </div>
          <div className="results-controls">
            <label className="field wide">
              <span>sweep</span>
              <select value={sweepPath} onChange={(event) => setSweepPath(event.target.value)}>
                <option value="">Select</option>
                {(sweeps.data ?? []).map((item) => (
                  <option key={item.path} value={item.path}>
                    {item.name} ({item.trial_count})
                  </option>
                ))}
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
              <select value={metricGoal} onChange={(event) => setMetricGoal(event.target.value as MetricGoal)}>
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
          </div>
          {(sweeps.error || sweepSummary.error) && <div className="error-state">{(sweeps.error ?? sweepSummary.error)?.message}</div>}
          {!sweepPath ? (
            <div className="empty-state">No sweep results</div>
          ) : (
            <SweepResults summary={sweepSummary.data} />
          )}
        </section>
      </div>
    </main>
  );
}

function RunResults({ run, groupRuns }: { run: ExperimentResult; groupRuns: ExperimentResult[] }) {
  const runs = groupRuns.length ? groupRuns : [run];
  const trainPoints = aggregateHistoryPoints(runs.map((item) => historyPoints(item.train_history, "return")));
  const lossPoints = aggregateHistoryPoints(runs.map((item) => historyPoints(item.train_history, "loss")));
  const evalPoints = aggregateHistoryPoints(runs.map((item) => historyPoints(item.eval_history, "return")));
  const isSeedGroup = runs.length > 1;
  return (
    <>
      <div className="result-heading">
        <div>
          <h2>{run.experiment_id}</h2>
          <span>{run.workflow_name}</span>
        </div>
        <span className="status-pill">{run.status}</span>
      </div>
      <MetricStrip
        metrics={[
          ["mean train", run.metrics.mean_train_return],
          ["last 10", run.metrics.mean_train_return_last_10],
          ["mean eval", run.metrics.mean_eval_return],
          ["episodes", run.metrics.train_episodes],
          ["seeds", isSeedGroup ? runs.length : null],
        ]}
      />
      <LinePlot id="train-return-plot" title={isSeedGroup ? "Training Return Mean" : "Training Return"} xLabel="Episode" yLabel="Return" points={trainPoints} />
      {lossPoints.length > 0 && <LinePlot id="train-loss-plot" title={isSeedGroup ? "Training Loss Mean" : "Training Loss"} xLabel="Episode" yLabel="Loss" points={lossPoints} />}
      {evalPoints.length > 0 && <LinePlot id="eval-return-plot" title={isSeedGroup ? "Evaluation Return Mean" : "Evaluation Return"} xLabel="Episode" yLabel="Return" points={evalPoints} />}
    </>
  );
}

function SweepResults({ summary }: { summary?: SweepSummary }) {
  if (!summary) return <div className="empty-state">Select a completed sweep</div>;
  return (
    <>
      <MetricStrip
        metrics={[
          ["best", summary.best?.metric],
          ["groups", summary.groups.length],
          ["trials", summary.trials.length],
          ["metric", summary.metric],
        ]}
      />
      <SweepRankPlot id="sweep-rank-plot" summary={summary} />
      <div className="table-scroll publication-table">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Mean</th>
              <th>Min</th>
              <th>Max</th>
              <th>Seeds</th>
              <th>Parameters</th>
            </tr>
          </thead>
          <tbody>
            {summary.groups.slice(0, 20).map((group, index) => (
              <tr key={group.group_id}>
                <td>{index + 1}</td>
                <td>{formatMetric(group.metric_mean)}</td>
                <td>{formatMetric(group.metric_min)}</td>
                <td>{formatMetric(group.metric_max)}</td>
                <td>{group.metric_count}</td>
                <td>{formatParameters(group.parameters)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function MetricStrip({ metrics }: { metrics: Array<[string, unknown]> }) {
  return (
    <div className="summary-strip paper-strip">
      {metrics.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{formatMetric(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function LinePlot({
  id,
  title,
  xLabel,
  yLabel,
  points,
}: {
  id: string;
  title: string;
  xLabel: string;
  yLabel: string;
  points: Array<{ x: number; y: number }>;
}) {
  if (points.length === 0) return null;
  const frame = plotFrame(points);
  const polyline = points.map((point) => `${xScale(point.x, frame)},${yScale(point.y, frame)}`).join(" ");
  return (
    <figure className="publication-plot">
      <div className="plot-header">
        <div className="plot-title">
          <LineChart size={16} />
          {title}
        </div>
        <button onClick={() => downloadSvg(id, `${id}.svg`)}>
          <Download size={14} />
          SVG
        </button>
      </div>
      <svg id={id} viewBox="0 0 760 340" role="img" aria-label={title}>
        <PlotAxes frame={frame} xLabel={xLabel} yLabel={yLabel} />
        <polyline points={polyline} fill="none" stroke="#0f5f6f" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    </figure>
  );
}

function SweepRankPlot({ id, summary }: { id: string; summary: SweepSummary }) {
  const points = summary.groups.slice(0, 20).map((group, index) => ({ x: index + 1, y: group.metric_mean }));
  if (points.length === 0) return <div className="empty-state">No completed sweep trials</div>;
  const frame = plotFrame([
    ...points,
    ...summary.groups.slice(0, 20).map((group, index) => ({ x: index + 1, y: group.metric_min })),
    ...summary.groups.slice(0, 20).map((group, index) => ({ x: index + 1, y: group.metric_max })),
  ]);
  return (
    <figure className="publication-plot">
      <div className="plot-header">
        <div className="plot-title">
          <BarChart3 size={16} />
          Sweep Ranking
        </div>
        <button onClick={() => downloadSvg(id, `${id}.svg`)}>
          <Download size={14} />
          SVG
        </button>
      </div>
      <svg id={id} viewBox="0 0 760 340" role="img" aria-label="Sweep ranking">
        <PlotAxes frame={frame} xLabel="Rank" yLabel={summary.metric} />
        {summary.groups.slice(0, 20).map((group, index) => (
          <SweepPoint key={group.group_id} group={group} rank={index + 1} frame={frame} />
        ))}
      </svg>
    </figure>
  );
}

function SweepPoint({ group, rank, frame }: { group: SweepGroupSummary; rank: number; frame: PlotFrame }) {
  const x = xScale(rank, frame);
  const y = yScale(group.metric_mean, frame);
  const yMin = yScale(group.metric_min, frame);
  const yMax = yScale(group.metric_max, frame);
  return (
    <g>
      <line x1={x} y1={yMin} x2={x} y2={yMax} stroke="#8aa0ae" strokeWidth="1.4" />
      <circle cx={x} cy={y} r="4.5" fill="#0f5f6f" />
    </g>
  );
}

type PlotFrame = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

function PlotAxes({ frame, xLabel, yLabel }: { frame: PlotFrame; xLabel: string; yLabel: string }) {
  const left = frame.margin.left;
  const right = frame.width - frame.margin.right;
  const top = frame.margin.top;
  const bottom = frame.height - frame.margin.bottom;
  const xTicks = ticks(frame.xMin, frame.xMax, 5);
  const yTicks = ticks(frame.yMin, frame.yMax, 5);
  return (
    <g>
      <rect x="0" y="0" width={frame.width} height={frame.height} fill="#ffffff" />
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line x1={left} y1={yScale(tick, frame)} x2={right} y2={yScale(tick, frame)} stroke="#e3e8ee" />
          <text x={left - 10} y={yScale(tick, frame) + 4} textAnchor="end" fontSize="11" fill="#334155">
            {formatMetric(tick)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line x1={xScale(tick, frame)} y1={bottom} x2={xScale(tick, frame)} y2={bottom + 5} stroke="#334155" />
          <text x={xScale(tick, frame)} y={bottom + 20} textAnchor="middle" fontSize="11" fill="#334155">
            {formatMetric(tick)}
          </text>
        </g>
      ))}
      <line x1={left} y1={top} x2={left} y2={bottom} stroke="#1f2937" strokeWidth="1.2" />
      <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="#1f2937" strokeWidth="1.2" />
      <text x={(left + right) / 2} y={frame.height - 10} textAnchor="middle" fontSize="12" fill="#1f2937">
        {xLabel}
      </text>
      <text
        x="16"
        y={(top + bottom) / 2}
        textAnchor="middle"
        fontSize="12"
        fill="#1f2937"
        transform={`rotate(-90 16 ${(top + bottom) / 2})`}
      >
        {yLabel}
      </text>
    </g>
  );
}

function plotFrame(points: Array<{ x: number; y: number }>): PlotFrame {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const yPad = yMinRaw === yMaxRaw ? Math.max(1, Math.abs(yMinRaw) * 0.1) : (yMaxRaw - yMinRaw) * 0.08;
  return {
    width: 760,
    height: 340,
    margin: { top: 20, right: 24, bottom: 48, left: 64 },
    xMin: xMin === xMax ? xMin - 1 : xMin,
    xMax: xMin === xMax ? xMax + 1 : xMax,
    yMin: yMinRaw - yPad,
    yMax: yMaxRaw + yPad,
  };
}

function xScale(value: number, frame: PlotFrame): number {
  const left = frame.margin.left;
  const width = frame.width - frame.margin.left - frame.margin.right;
  return left + ((value - frame.xMin) / (frame.xMax - frame.xMin)) * width;
}

function yScale(value: number, frame: PlotFrame): number {
  const top = frame.margin.top;
  const height = frame.height - frame.margin.top - frame.margin.bottom;
  return top + (1 - (value - frame.yMin) / (frame.yMax - frame.yMin)) * height;
}

function ticks(min: number, max: number, count: number): number[] {
  if (count <= 1) return [min];
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function historyPoints(history: ExperimentHistoryPoint[], key: "return" | "loss"): Array<{ x: number; y: number }> {
  return history
    .map((point) => ({ x: Number(point.episode), y: numberValue(point[key]) }))
    .filter((point): point is { x: number; y: number } => Number.isFinite(point.x) && point.y !== null);
}

function aggregateHistoryPoints(seriesList: Array<Array<{ x: number; y: number }>>): Array<{ x: number; y: number }> {
  const valuesByEpisode = new Map<number, number[]>();
  for (const series of seriesList) {
    for (const point of series) {
      const values = valuesByEpisode.get(point.x) ?? [];
      values.push(point.y);
      valuesByEpisode.set(point.x, values);
    }
  }
  return Array.from(valuesByEpisode.entries())
    .sort(([left], [right]) => left - right)
    .map(([x, values]) => ({ x, y: values.reduce((total, value) => total + value, 0) / values.length }));
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatMetric(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  if (Math.abs(value) >= 100 || Number.isInteger(value)) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toPrecision(4);
}

function formatParameters(parameters: Record<string, unknown>): string {
  return Object.entries(parameters)
    .map(([key, value]) => `${key}=${formatMetric(value) || String(value)}`)
    .join(", ");
}

function intValue(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function downloadSvg(id: string, filename: string) {
  const svg = document.getElementById(id);
  if (!svg) return;
  const blob = new Blob([svg.outerHTML], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
