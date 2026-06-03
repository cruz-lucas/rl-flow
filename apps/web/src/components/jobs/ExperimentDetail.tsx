import { useQuery } from "@tanstack/react-query";
import { BarChart3, Download, FlaskConical, LineChart, Plus, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { ExperimentHistoryPoint, ExperimentResult, SweepGroupSummary, SweepSummary } from "../../types/schema";

type MetricGoal = "maximize" | "minimize";
type RunOptionKind = "run" | "sweep_group" | "sweep_trial";
type RunOption = {
  id: string;
  label: string;
  searchText: string;
  kind: RunOptionKind;
  run: ExperimentResult;
  groupRuns: ExperimentResult[];
};
type PlotSeries = {
  id: string;
  label: string;
  color: string;
  points: Array<{ x: number; y: number }>;
  band: Array<{ x: number; yLow: number; yHigh: number }>;
};

const plotColors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#6b4c9a", "#8a6d00", "#56b4e9", "#111827"];

export function ExperimentDetail() {
  const results = useQuery({ queryKey: ["experiment-results"], queryFn: api.experimentResults });
  const sweeps = useQuery({ queryKey: ["sweeps"], queryFn: api.sweeps });
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [hasSeededRunSelection, setHasSeededRunSelection] = useState(false);
  const [runSearch, setRunSearch] = useState("");
  const [runCandidateId, setRunCandidateId] = useState("");
  const [sweepPath, setSweepPath] = useState("");
  const [metricName, setMetricName] = useState("mean_eval_return");
  const [metricGoal, setMetricGoal] = useState<MetricGoal>("maximize");
  const [metricLastN, setMetricLastN] = useState(50);

  const runOptions = useMemo(() => buildRunOptions(results.data ?? []), [results.data]);
  const runOptionMap = useMemo(() => new Map(runOptions.map((option) => [option.id, option])), [runOptions]);
  const selectedRunOptions = useMemo(
    () => selectedRunIds.map((id) => runOptionMap.get(id)).filter((option): option is RunOption => Boolean(option)),
    [runOptionMap, selectedRunIds],
  );
  const filteredRunOptions = useMemo(
    () => filterRunOptions(runOptions, runSearch, selectedRunIds).slice(0, 100),
    [runOptions, runSearch, selectedRunIds],
  );

  useEffect(() => {
    if (runOptions.length === 0) {
      if (selectedRunIds.length > 0) setSelectedRunIds([]);
      if (hasSeededRunSelection) setHasSeededRunSelection(false);
      return;
    }
    if (selectedRunIds.length === 0) {
      if (!hasSeededRunSelection) {
        setSelectedRunIds([runOptions[0].id]);
        setHasSeededRunSelection(true);
      }
      return;
    }
    const validIds = new Set(runOptions.map((option) => option.id));
    const keptIds = selectedRunIds.filter((id) => validIds.has(id));
    const nextIds = keptIds.length > 0 ? keptIds : [runOptions[0].id];
    if (!sameStringArray(selectedRunIds, nextIds)) {
      setSelectedRunIds(nextIds);
    }
  }, [hasSeededRunSelection, runOptions, selectedRunIds]);

  useEffect(() => {
    if (runCandidateId && filteredRunOptions.some((option) => option.id === runCandidateId)) {
      return;
    }
    setRunCandidateId(filteredRunOptions[0]?.id ?? "");
  }, [filteredRunOptions, runCandidateId]);

  useEffect(() => {
    if (!sweepPath && sweeps.data?.[0]) {
      setSweepPath(sweeps.data[0].path);
    }
  }, [sweepPath, sweeps.data]);

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
          </div>
          <div className="run-add-controls">
            <label className="field wide">
              <span>search runs</span>
              <input value={runSearch} onChange={(event) => setRunSearch(event.target.value)} placeholder="group, trial, seed, parameter, path" />
            </label>
            <label className="field wide">
              <span>match</span>
              <select value={runCandidateId} onChange={(event) => setRunCandidateId(event.target.value)}>
                <option value="">No matches</option>
                {filteredRunOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={() => {
                if (!runCandidateId || selectedRunIds.includes(runCandidateId)) return;
                setSelectedRunIds((ids) => [...ids, runCandidateId]);
              }}
              disabled={!runCandidateId}
            >
              <Plus size={16} />
              Add
            </button>
            <button onClick={() => setSelectedRunIds([])} disabled={selectedRunIds.length === 0}>
              <X size={16} />
              Clear
            </button>
          </div>
          {results.error && <div className="error-state">{results.error.message}</div>}
          {runOptions.length === 0 ? (
            <div className="empty-state">No run results</div>
          ) : selectedRunOptions.length === 0 ? (
            <div className="empty-state">Add runs, sweep groups, or individual trials to compare</div>
          ) : (
            <RunResults
              options={selectedRunOptions}
              onRemove={(id) => setSelectedRunIds((ids) => ids.filter((selectedId) => selectedId !== id))}
            />
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
            <SweepResults summary={sweepSummary.data} experimentResults={results.data ?? []} />
          )}
        </section>
      </div>
    </main>
  );
}

function RunResults({ options, onRemove }: { options: RunOption[]; onRemove: (id: string) => void }) {
  const trainSeries = comparisonSeries(options, "train", "return", "env_step");
  const lossSeries = comparisonSeries(options, "train", "loss", "env_step");
  const evalSeries = comparisonSeries(options, "eval", "return", "env_step");
  return (
    <>
      <div className="result-heading">
        <div>
          <h2>Run Comparison</h2>
          <span>{options.length} selected</span>
        </div>
        <span className="status-pill">{options.length === 1 ? optionTypeLabel(options[0]) : "compare"}</span>
      </div>
      <MetricStrip
        metrics={[
          ["selected", options.length],
          ["groups", options.filter((option) => option.kind === "sweep_group").length],
          ["trials", options.filter((option) => option.kind === "sweep_trial").length],
          ["runs", options.filter((option) => option.kind === "run").length],
        ]}
      />
      <RunComparisonTable options={options} onRemove={onRemove} />
      <MultiLinePlot id="train-return-plot" title="Training Return" xLabel="Environment Steps" yLabel="Return" series={trainSeries} />
      {lossSeries.length > 0 && <MultiLinePlot id="train-loss-plot" title="Training Loss" xLabel="Environment Steps" yLabel="Loss" series={lossSeries} />}
      {evalSeries.length > 0 && <MultiLinePlot id="eval-return-plot" title="Evaluation Return" xLabel="Environment Steps" yLabel="Return" series={evalSeries} />}
    </>
  );
}

function RunComparisonTable({ options, onRemove }: { options: RunOption[]; onRemove: (id: string) => void }) {
  return (
    <div className="table-scroll comparison-table">
      <table className="data-table">
        <thead>
          <tr>
            <th />
            <th>Run</th>
            <th>Type</th>
            <th>Status</th>
            <th>Seeds</th>
            <th>Mean Train</th>
            <th>Last 10</th>
            <th>Mean Eval</th>
            <th>Parameters</th>
          </tr>
        </thead>
        <tbody>
          {options.map((option) => (
            <tr key={option.id}>
              <td>
                <button className="icon-button" onClick={() => onRemove(option.id)} aria-label={`Remove ${option.label}`}>
                  <X size={14} />
                </button>
              </td>
              <td>
                <strong>{shortRunLabel(option)}</strong>
                <span className="muted-line">{option.run.workflow_name}</span>
                {option.run.sweep_id && <span className="muted-line">{sweepFolderName(option.run)}</span>}
              </td>
              <td>{optionTypeLabel(option)}</td>
              <td>{statusLabel(option.groupRuns)}</td>
              <td>{option.groupRuns.length}</td>
              <td>{formatMetric(metricMean(option.groupRuns, "mean_train_return"))}</td>
              <td>{formatMetric(metricMean(option.groupRuns, "mean_train_return_last_10"))}</td>
              <td>{formatMetric(metricMean(option.groupRuns, "mean_eval_return"))}</td>
              <td>{formatParameters(option.run.sweep_group_parameters ?? option.run.sweep_parameters)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SweepResults({ summary, experimentResults }: { summary?: SweepSummary; experimentResults: ExperimentResult[] }) {
  if (!summary) return <div className="empty-state">Select a completed sweep</div>;
  const bestRuns = runsForBestGroup(summary, experimentResults);
  return (
    <>
      <MetricStrip
        metrics={[
          ["best group", summary.best?.group_id],
          ["best metric", summary.best?.metric],
          ["groups", summary.groups.length],
          ["trials", summary.trials.length],
          ["best seeds", summary.best?.metric_count],
          ["metric", summary.metric],
        ]}
      />
      <SweepRankPlot id="sweep-rank-plot" summary={summary} />
      <BestGroupLearningCurve summary={summary} runs={bestRuns} />
      <div className="table-scroll publication-table">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Group</th>
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
                <td>{group.group_id}</td>
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

function BestGroupLearningCurve({ summary, runs }: { summary: SweepSummary; runs: ExperimentResult[] }) {
  if (!summary.best || runs.length === 0) return null;
  const series = seriesFromRuns({
    id: `best:${summary.best.group_id}`,
    label: `${summary.best.group_id} mean`,
    color: plotColors[0],
    runs,
    history: "train",
    key: "return",
  });
  if (series.points.length === 0) return null;
  const title =
    runs.length > 1
      ? `Best Group ${summary.best.group_id} Training Return Mean`
      : `Best Group ${summary.best.group_id} Training Return`;
  return <MultiLinePlot id="best-group-train-return-plot" title={title} xLabel="Episode" yLabel="Return" series={[series]} />;
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
  const clipId = `${id}-clip`;
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
      <svg id={id} viewBox={`0 0 ${frame.width} ${frame.height}`} role="img" aria-label={title} fontFamily="Inter, Arial, sans-serif">
        <defs>
          <clipPath id={clipId}>
            <rect x={frame.margin.left} y={frame.margin.top} width={plotWidth(frame)} height={plotHeight(frame)} />
          </clipPath>
        </defs>
        <PlotAxes frame={frame} xLabel={xLabel} yLabel={yLabel} />
        <text x={frame.width / 2} y="24" textAnchor="middle" fontSize="15" fontWeight="700" fill="#111827">
          {title}
        </text>
        <polyline
          points={polyline}
          clipPath={`url(#${clipId})`}
          fill="none"
          stroke={plotColors[0]}
          strokeWidth="2.6"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
    </figure>
  );
}

function MultiLinePlot({
  id,
  title,
  xLabel,
  yLabel,
  series,
}: {
  id: string;
  title: string;
  xLabel: string;
  yLabel: string;
  series: PlotSeries[];
}) {
  const visibleSeries = series.filter((item) => item.points.length > 0);
  if (visibleSeries.length === 0) return null;
  const legendColumns = visibleSeries.length === 1 ? 1 : 2;
  const legendRows = Math.ceil(visibleSeries.length / legendColumns);
  const frame = plotFrame(
    visibleSeries.flatMap((item) => item.points),
    visibleSeries.flatMap((item) => item.band),
    { legendRows },
  );
  const clipId = `${id}-clip`;
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
      <svg id={id} viewBox={`0 0 ${frame.width} ${frame.height}`} role="img" aria-label={title} fontFamily="Inter, Arial, sans-serif">
        <defs>
          <clipPath id={clipId}>
            <rect x={frame.margin.left} y={frame.margin.top} width={plotWidth(frame)} height={plotHeight(frame)} />
          </clipPath>
        </defs>
        <PlotAxes frame={frame} xLabel={xLabel} yLabel={yLabel} />
        <text x={frame.width / 2} y="24" textAnchor="middle" fontSize="15" fontWeight="700" fill="#111827">
          {title}
        </text>
        {visibleSeries.map((item) =>
          item.band.length > 1 ? (
            <path
              key={`${item.id}:band`}
              d={bandPath(item.band, frame)}
              clipPath={`url(#${clipId})`}
              fill={item.color}
              opacity="0.14"
            />
          ) : null,
        )}
        {visibleSeries.map((item) => (
          <polyline
            key={item.id}
            points={item.points.map((point) => `${xScale(point.x, frame)},${yScale(point.y, frame)}`).join(" ")}
            clipPath={`url(#${clipId})`}
            fill="none"
            stroke={item.color}
            strokeWidth="2.6"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        <PlotLegend series={visibleSeries} frame={frame} columns={legendColumns} />
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
  const clipId = `${id}-clip`;
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
      <svg id={id} viewBox={`0 0 ${frame.width} ${frame.height}`} role="img" aria-label="Sweep ranking" fontFamily="Inter, Arial, sans-serif">
        <defs>
          <clipPath id={clipId}>
            <rect x={frame.margin.left} y={frame.margin.top} width={plotWidth(frame)} height={plotHeight(frame)} />
          </clipPath>
        </defs>
        <PlotAxes frame={frame} xLabel="Rank" yLabel={summary.metric} />
        <text x={frame.width / 2} y="24" textAnchor="middle" fontSize="15" fontWeight="700" fill="#111827">
          Sweep Ranking
        </text>
        {summary.groups.slice(0, 20).map((group, index) => (
          <SweepPoint key={group.group_id} group={group} rank={index + 1} frame={frame} clipId={clipId} />
        ))}
      </svg>
    </figure>
  );
}

function SweepPoint({ group, rank, frame, clipId }: { group: SweepGroupSummary; rank: number; frame: PlotFrame; clipId: string }) {
  const x = xScale(rank, frame);
  const y = yScale(group.metric_mean, frame);
  const yMin = yScale(group.metric_min, frame);
  const yMax = yScale(group.metric_max, frame);
  return (
    <g clipPath={`url(#${clipId})`}>
      <line x1={x} y1={yMin} x2={x} y2={yMax} stroke="#606f7b" strokeWidth="1.6" />
      <line x1={x - 5} y1={yMin} x2={x + 5} y2={yMin} stroke="#606f7b" strokeWidth="1.6" />
      <line x1={x - 5} y1={yMax} x2={x + 5} y2={yMax} stroke="#606f7b" strokeWidth="1.6" />
      <circle cx={x} cy={y} r="4.8" fill={plotColors[0]} stroke="#ffffff" strokeWidth="1.2" />
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
  legendRows: number;
  legendHeight: number;
};

function PlotAxes({ frame, xLabel, yLabel }: { frame: PlotFrame; xLabel: string; yLabel: string }) {
  const left = frame.margin.left;
  const right = frame.width - frame.margin.right;
  const top = frame.margin.top;
  const bottom = frame.height - frame.margin.bottom;
  const xTicks = niceTicks(frame.xMin, frame.xMax, 6, true);
  const yTicks = niceTicks(frame.yMin, frame.yMax, 6, false);
  const xLabelY = frame.height - frame.legendHeight - 18;
  return (
    <g>
      <rect x="0" y="0" width={frame.width} height={frame.height} fill="#ffffff" />
      <rect x={left} y={top} width={right - left} height={bottom - top} fill="#ffffff" />
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line x1={left} y1={yScale(tick, frame)} x2={right} y2={yScale(tick, frame)} stroke="#e6ebf0" strokeWidth="1" />
          <text x={left - 12} y={yScale(tick, frame) + 4} textAnchor="end" fontSize="12" fill="#111827">
            {formatTick(tick)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line x1={xScale(tick, frame)} y1={bottom} x2={xScale(tick, frame)} y2={bottom + 6} stroke="#111827" strokeWidth="1.1" />
          <text x={xScale(tick, frame)} y={bottom + 24} textAnchor="middle" fontSize="12" fill="#111827">
            {formatTick(tick)}
          </text>
        </g>
      ))}
      <line x1={left} y1={top} x2={left} y2={bottom} stroke="#111827" strokeWidth="1.4" />
      <line x1={left} y1={bottom} x2={right} y2={bottom} stroke="#111827" strokeWidth="1.4" />
      <text x={(left + right) / 2} y={xLabelY} textAnchor="middle" fontSize="13" fontWeight="600" fill="#111827">
        {xLabel}
      </text>
      <text
        x="22"
        y={(top + bottom) / 2}
        textAnchor="middle"
        fontSize="13"
        fontWeight="600"
        fill="#111827"
        transform={`rotate(-90 22 ${(top + bottom) / 2})`}
      >
        {yLabel}
      </text>
    </g>
  );
}

function PlotLegend({ series, frame, columns }: { series: PlotSeries[]; frame: PlotFrame; columns: number }) {
  if (frame.legendRows === 0) return null;
  const left = frame.margin.left;
  const columnWidth = plotWidth(frame) / columns;
  const startY = frame.height - frame.legendHeight + 18;
  return (
    <g>
      {series.map((item, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const x = left + column * columnWidth;
        const y = startY + row * 22;
        return (
          <g key={`legend-${item.id}`}>
            <line x1={x} y1={y - 4} x2={x + 24} y2={y - 4} stroke={item.color} strokeWidth="2.8" strokeLinecap="round" />
            {item.band.length > 1 && <rect x={x} y={y - 8} width="24" height="8" fill={item.color} opacity="0.14" />}
            <text x={x + 32} y={y} fontSize="12" fill="#111827">
              {truncateLabel(item.label, columns === 1 ? 78 : 38)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function plotFrame(
  points: Array<{ x: number; y: number }>,
  bands: Array<{ x: number; yLow: number; yHigh: number }> = [],
  options: { legendRows?: number } = {},
): PlotFrame {
  const xs = points.map((point) => point.x);
  const ys = [...points.map((point) => point.y), ...bands.flatMap((point) => [point.yLow, point.yHigh])];
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const xPad = xMin === xMax ? Math.max(1, Math.abs(xMin) * 0.05) : 0;
  const yPad = yMinRaw === yMaxRaw ? Math.max(1, Math.abs(yMinRaw) * 0.1) : (yMaxRaw - yMinRaw) * 0.08;
  const yMinPadded = yMinRaw >= 0 && yMinRaw - yPad >= -0.05 * Math.max(1, yMaxRaw) ? 0 : yMinRaw - yPad;
  const legendRows = options.legendRows ?? 0;
  const legendHeight = legendRows > 0 ? legendRows * 22 + 12 : 0;
  const baseHeight = 420;
  return {
    width: 820,
    height: baseHeight + legendHeight,
    margin: { top: 54, right: 32, bottom: 62 + legendHeight, left: 84 },
    xMin: xMin === xMax ? xMin - xPad : xMin,
    xMax: xMin === xMax ? xMax + xPad : xMax,
    yMin: yMinPadded,
    yMax: yMaxRaw + yPad,
    legendRows,
    legendHeight,
  };
}

function xScale(value: number, frame: PlotFrame): number {
  const left = frame.margin.left;
  const width = frame.width - frame.margin.left - frame.margin.right;
  return left + ((value - frame.xMin) / (frame.xMax - frame.xMin)) * width;
}

function yScale(value: number, frame: PlotFrame): number {
  const top = frame.margin.top;
  const height = plotHeight(frame);
  return top + (1 - (value - frame.yMin) / (frame.yMax - frame.yMin)) * height;
}

function plotWidth(frame: PlotFrame): number {
  return frame.width - frame.margin.left - frame.margin.right;
}

function plotHeight(frame: PlotFrame): number {
  return frame.height - frame.margin.top - frame.margin.bottom;
}

function niceTicks(min: number, max: number, count: number, integersOnly: boolean): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || count <= 1) return [];
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count - 1);
  const step = niceStep(rawStep, integersOnly);
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 0.5; value += step) {
    const normalized = Math.abs(value) < step * 1e-6 ? 0 : value;
    ticks.push(Number(normalized.toPrecision(12)));
  }
  if (integersOnly && Number.isInteger(min) && ticks[0] !== min) {
    ticks.unshift(min);
  }
  if (integersOnly && Number.isInteger(max) && ticks[ticks.length - 1] !== max) {
    ticks.push(max);
  }
  return ticks.length ? ticks : [min, max];
}

function niceStep(rawStep: number, integersOnly: boolean): number {
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(rawStep));
  const fraction = rawStep / power;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const step = niceFraction * power;
  return integersOnly ? Math.max(1, Math.round(step)) : step;
}

function formatTick(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1000) return formatCompactTick(value);
  if (absolute > 0 && absolute < 0.001) return value.toExponential(1);
  if (absolute >= 10 || Number.isInteger(value)) return value.toFixed(0);
  if (absolute >= 1) return value.toFixed(1).replace(/\.0$/, "");
  return value.toPrecision(2);
}

function formatCompactTick(value: number): string {
  const absolute = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (absolute >= 1000000) {
    return `${sign}${trimTrailingZeros(absolute / 1000000)}M`;
  }
  return `${sign}${trimTrailingZeros(absolute / 1000)}k`;
}

function trimTrailingZeros(value: number): string {
  return value.toFixed(value >= 10 ? 0 : 1).replace(/\.0$/, "");
}

function bandPath(band: Array<{ x: number; yLow: number; yHigh: number }>, frame: PlotFrame): string {
  const upper = band.map((point) => `${xScale(point.x, frame)},${yScale(point.yHigh, frame)}`).join(" L ");
  const lower = [...band].reverse().map((point) => `${xScale(point.x, frame)},${yScale(point.yLow, frame)}`).join(" L ");
  return `M ${upper} L ${lower} Z`;
}

function historyPoints(history: ExperimentHistoryPoint[], key: "return" | "loss", xKey: "episode" | "env_step" = "episode"): Array<{ x: number; y: number }> {
  return history
    .map((point) => ({ x: xValue(point, xKey), y: numberValue(point[key]) }))
    .filter((point): point is { x: number; y: number } => Number.isFinite(point.x) && point.y !== null);
}

function comparisonSeries(options: RunOption[], history: "train" | "eval", key: "return" | "loss", xKey: "episode" | "env_step"): PlotSeries[] {
  return options.map((option, index) =>
    seriesFromRuns({
      id: `${option.id}:${history}:${key}`,
      label: shortRunLabel(option),
      color: plotColors[index % plotColors.length],
      runs: option.groupRuns,
      history,
      key,
      xKey,
    }),
  );
}

function seriesFromRuns({
  id,
  label,
  color,
  runs,
  history,
  key,
  xKey = "episode",
}: {
  id: string;
  label: string;
  color: string;
  runs: ExperimentResult[];
  history: "train" | "eval";
  key: "return" | "loss";
  xKey?: "episode" | "env_step";
}): PlotSeries {
  const histories = runs.map((run) => (history === "train" ? run.train_history : run.eval_history));
  const stats = aggregateHistoryStats(histories.map((items) => historyPoints(items, key, xKey)));
  return {
    id,
    label,
    color,
    points: stats.map((point) => ({ x: point.x, y: point.y })),
    band: stats.filter((point) => point.count > 1 && point.yLow !== point.yHigh).map((point) => ({ x: point.x, yLow: point.yLow, yHigh: point.yHigh })),
  };
}

function buildRunOptions(results: ExperimentResult[]): RunOption[] {
  const options: RunOption[] = [];
  const sweepGroups = new Map<string, RunOption>();

  for (const run of results) {
    if (!run.sweep_id) {
      options.push(makeRunOption(run));
      continue;
    }
    if (run.sweep_group_id) {
      const groupId = groupOptionId(run);
      let option = sweepGroups.get(groupId);
      if (!option) {
        option = makeGroupOption(run, []);
        sweepGroups.set(groupId, option);
        options.push(option);
      }
      option.groupRuns.push(run);
      option.label = groupOptionLabel(option.run, option.groupRuns);
      option.searchText = searchTextForOption(option);
    }
    options.push(makeTrialOption(run));
  }

  return options;
}

function makeRunOption(run: ExperimentResult): RunOption {
  const option = {
    id: `run:${run.run_dir}`,
    label: `Run: ${run.experiment_id}`,
    searchText: "",
    kind: "run" as const,
    run,
    groupRuns: [run],
  };
  return { ...option, searchText: searchTextForOption(option) };
}

function makeGroupOption(run: ExperimentResult, groupRuns: ExperimentResult[]): RunOption {
  const option = {
    id: groupOptionId(run),
    label: groupOptionLabel(run, groupRuns),
    searchText: "",
    kind: "sweep_group" as const,
    run,
    groupRuns,
  };
  return { ...option, searchText: searchTextForOption(option) };
}

function makeTrialOption(run: ExperimentResult): RunOption {
  const option = {
    id: `trial:${run.run_dir}`,
    label: trialOptionLabel(run),
    searchText: "",
    kind: "sweep_trial" as const,
    run,
    groupRuns: [run],
  };
  return { ...option, searchText: searchTextForOption(option) };
}

function groupOptionId(run: ExperimentResult): string {
  return `group:${run.sweep_group_run_dir ?? `${run.sweep_dir ?? run.sweep_id}:${run.sweep_group_id}`}`;
}

function groupOptionLabel(run: ExperimentResult, runs: ExperimentResult[]): string {
  const parameters = formatParameters(run.sweep_group_parameters ?? run.sweep_parameters);
  const suffix = parameters ? `, ${parameters}` : "";
  return `Group: ${sweepFolderName(run)} / ${run.sweep_group_id} (${runs.length} seeds${suffix})`;
}

function trialOptionLabel(run: ExperimentResult): string {
  const seed = run.seed !== null && run.seed !== undefined ? ` / seed ${String(run.seed)}` : "";
  const parameters = formatParameters(run.sweep_parameters);
  const details = [run.sweep_group_id, parameters].filter(Boolean).join(", ");
  const suffix = details ? ` (${details})` : "";
  return `Trial: ${sweepFolderName(run)} / ${run.sweep_trial_id ?? run.experiment_id}${seed}${suffix}`;
}

function filterRunOptions(options: RunOption[], query: string, selectedIds: string[]): RunOption[] {
  const selected = new Set(selectedIds);
  const terms = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  return options.filter((option) => {
    if (selected.has(option.id)) return false;
    if (terms.length === 0) return true;
    return terms.every((term) => option.searchText.includes(term));
  });
}

function searchTextForOption(option: Omit<RunOption, "searchText">): string {
  const run = option.run;
  return [
    option.label,
    option.id,
    option.kind,
    optionTypeLabel(option),
    shortRunLabel(option),
    run.experiment_id,
    run.workflow_name,
    run.run_dir,
    run.sweep_dir,
    run.sweep_id,
    run.sweep_trial_id,
    run.sweep_group_id,
    run.sweep_group_run_dir,
    run.seed,
    JSON.stringify(run.sweep_parameters),
    JSON.stringify(run.sweep_group_parameters ?? {}),
  ]
    .filter((value) => value !== null && value !== undefined)
    .join(" ")
    .toLowerCase();
}

function shortRunLabel(option: Pick<RunOption, "kind" | "run" | "groupRuns">): string {
  const run = option.run;
  if (option.kind === "sweep_group") {
    return `${sweepFolderName(run)} ${run.sweep_group_id} mean`;
  }
  if (option.kind === "sweep_trial") {
    const seed = run.seed !== null && run.seed !== undefined ? ` seed ${String(run.seed)}` : "";
    return `${sweepFolderName(run)} ${run.sweep_trial_id ?? run.experiment_id}${seed}`;
  }
  return run.experiment_id;
}

function optionTypeLabel(option: Pick<RunOption, "kind" | "groupRuns">): string {
  if (option.kind === "sweep_group") return "seed group";
  if (option.kind === "sweep_trial") return "trial";
  return "run";
}

function sweepFolderName(run: ExperimentResult): string {
  const path = run.sweep_dir ?? run.sweep_group_run_dir?.split("/trials/")[0] ?? run.run_dir.split("/trials/")[0] ?? run.sweep_id;
  if (!path) return run.sweep_id ?? "sweep";
  const normalized = path.replace(/\/+$/, "");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || run.sweep_id || "sweep";
}

function runsForBestGroup(summary: SweepSummary, results: ExperimentResult[]): ExperimentResult[] {
  const best = summary.best;
  if (!best) return [];
  const bestTrialIds = new Set(best.trial_ids);
  return results.filter((run) => {
    if (run.sweep_id !== summary.sweep_id) return false;
    if (run.sweep_group_id === best.group_id) return true;
    return typeof run.sweep_trial_id === "string" && bestTrialIds.has(run.sweep_trial_id);
  });
}

function metricMean(runs: ExperimentResult[], key: string): number | null {
  const values = runs.map((run) => numberValue(run.metrics[key])).filter((value): value is number => value !== null);
  if (values.length === 0) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function metricMax(runs: ExperimentResult[], key: string): number | null {
  const values = runs.map((run) => numberValue(run.metrics[key])).filter((value): value is number => value !== null);
  return values.length ? Math.max(...values) : null;
}

function statusLabel(runs: ExperimentResult[]): string {
  const statuses = Array.from(new Set(runs.map((run) => run.status)));
  return statuses.length === 1 ? statuses[0] : "mixed";
}

function sameStringArray(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function aggregateHistoryStats(seriesList: Array<Array<{ x: number; y: number }>>): Array<{ x: number; y: number; yLow: number; yHigh: number; count: number }> {
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
    .map(([x, values]) => ({
      x,
      y: values.reduce((total, value) => total + value, 0) / values.length,
      yLow: Math.min(...values),
      yHigh: Math.max(...values),
      count: values.length,
    }));
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function xValue(point: ExperimentHistoryPoint, xKey: "episode" | "env_step"): number {
  if (xKey === "env_step" && typeof point.env_step === "number" && Number.isFinite(point.env_step)) {
    return point.env_step;
  }
  return Number(point.episode);
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

function truncateLabel(value: string, length: number): string {
  return value.length <= length ? value : `${value.slice(0, Math.max(0, length - 1))}...`;
}

function downloadSvg(id: string, filename: string) {
  const svg = document.getElementById(id);
  if (!svg) return;
  const clone = svg.cloneNode(true) as SVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", clone.getAttribute("viewBox")?.split(" ")[2] ?? "820");
  clone.setAttribute("height", clone.getAttribute("viewBox")?.split(" ")[3] ?? "420");
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
