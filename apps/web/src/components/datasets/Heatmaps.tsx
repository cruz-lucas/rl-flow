import { Download } from "lucide-react";
import { useRef } from "react";
import type { RefObject } from "react";
import type { DatasetVisitation, OfflineRndPoint } from "../../types/schema";

type ValueMatrix = Array<Array<number | null>>;
type ActionValueMatrix = Array<Array<Array<number | null>>>;

interface StateHeatmapProps {
  title: string;
  values: ValueMatrix | number[][];
  validMask?: boolean[][];
  valueLabel: string;
  palette?: "count" | "bonus";
  annotate?: boolean;
}

interface ActionHeatmapProps {
  title: string;
  values: ActionValueMatrix | number[][][];
  validMask?: boolean[][];
  actionLabels?: string[];
  valueLabel: string;
  palette?: "count" | "bonus";
}

export function StateVisitationHeatmap({ visitation }: { visitation: DatasetVisitation }) {
  return (
    <StateHeatmap
      title="State Visitation"
      values={visitation.state_counts}
      validMask={visitation.valid_mask}
      valueLabel="count"
      palette="count"
      annotate
    />
  );
}

export function ActionVisitationHeatmap({ visitation }: { visitation: DatasetVisitation }) {
  return (
    <ActionHeatmap
      title="State-Action Visitation"
      values={visitation.state_action_counts}
      validMask={visitation.valid_mask}
      actionLabels={visitation.action_labels}
      valueLabel="count"
      palette="count"
    />
  );
}

export function StateHeatmap({
  title,
  values,
  validMask,
  valueLabel,
  palette = "bonus",
  annotate = false,
}: StateHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const height = values.length;
  const width = values[0]?.length ?? 0;
  const domain = valueDomain(flattenValues(values, validMask));
  const showLabels = annotate && height <= 16 && width <= 16;
  return (
    <section className="heatmap-panel">
      <PlotHeader title={title} svgRef={svgRef} />
      <svg ref={svgRef} className="heatmap-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {values.map((row, rowIndex) =>
          row.map((value, colIndex) => {
            const valid = validMask?.[rowIndex]?.[colIndex] ?? true;
            const fill = valid && value !== null ? colorForValue(Number(value), domain, palette) : "#ffffff";
            const label = value === null ? "" : formatValue(Number(value));
            return (
              <g key={`${rowIndex}-${colIndex}`}>
                <rect
                  x={colIndex}
                  y={rowIndex}
                  width={1}
                  height={1}
                  fill={fill}
                  stroke="#c8d1dc"
                  strokeWidth={0.018}
                >
                  <title>
                    row {rowIndex}, col {colIndex}: {label || "empty"} {valueLabel}
                  </title>
                </rect>
                {showLabels && value !== null && Number(value) > 0 && (
                  <text
                    x={colIndex + 0.5}
                    y={rowIndex + 0.58}
                    textAnchor="middle"
                    fontSize={label.length > 3 ? 0.2 : 0.27}
                    fill={textColor(fill)}
                  >
                    {label}
                  </text>
                )}
              </g>
            );
          }),
        )}
      </svg>
      <Legend domain={domain} palette={palette} valueLabel={valueLabel} />
    </section>
  );
}

export function ActionHeatmap({
  title,
  values,
  validMask,
  actionLabels = ["Up", "Down", "Left", "Right"],
  valueLabel,
  palette = "bonus",
}: ActionHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const height = values.length;
  const width = values[0]?.length ?? 0;
  const domain = valueDomain(flattenActionValues(values, validMask));
  return (
    <section className="heatmap-panel">
      <PlotHeader title={title} svgRef={svgRef} />
      <svg ref={svgRef} className="heatmap-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {values.map((row, rowIndex) =>
          row.map((actions, colIndex) => {
            const valid = validMask?.[rowIndex]?.[colIndex] ?? true;
            const polygons = cellPolygons(rowIndex, colIndex);
            return polygons.map((points, actionIndex) => {
              const value = actions[actionIndex] ?? null;
              const fill = valid && value !== null ? colorForValue(Number(value), domain, palette) : "#ffffff";
              const label = value === null ? "" : formatValue(Number(value));
              return (
                <polygon
                  key={`${rowIndex}-${colIndex}-${actionIndex}`}
                  points={points}
                  fill={fill}
                  stroke="#4a5565"
                  strokeWidth={0.014}
                >
                  <title>
                    row {rowIndex}, col {colIndex}, {actionLabels[actionIndex] ?? `action ${actionIndex}`}:{" "}
                    {label || "empty"} {valueLabel}
                  </title>
                </polygon>
              );
            });
          }),
        )}
      </svg>
      <Legend domain={domain} palette={palette} valueLabel={valueLabel} />
    </section>
  );
}

export function BonusScatter({
  points,
  title = "Learned Bonus vs Count-Based Bonus",
  yLabel = "learned bonus",
  showReference = true,
}: {
  points: OfflineRndPoint[];
  title?: string;
  yLabel?: string;
  showReference?: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const finitePoints = points.filter(
    (point) => Number.isFinite(point.count_bonus) && Number.isFinite(point.learned_bonus),
  );
  const width = 360;
  const height = 240;
  const margin = { left: 46, right: 12, top: 12, bottom: 38 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMin = 0;
  const xMax = 1;
  const learnedValues = finitePoints.map((point) => point.learned_bonus);
  const rawYMin = Math.min(0, ...learnedValues);
  const rawYMax = Math.max(1e-6, ...learnedValues);
  const yPadding = Math.abs(rawYMax - rawYMin) < 1e-9 ? 1 : 0;
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;
  const x = (value: number) =>
    margin.left + ((Math.max(xMin, Math.min(xMax, value)) - xMin) / (xMax - xMin)) * plotWidth;
  const y = (value: number) =>
    margin.top +
    plotHeight -
    ((Math.max(yMin, Math.min(yMax, value)) - yMin) / (yMax - yMin)) * plotHeight;

  return (
    <section className="heatmap-panel">
      <PlotHeader title={title} svgRef={svgRef} />
      <svg ref={svgRef} className="scatter-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + plotHeight} stroke="#64748b" />
        <line
          x1={margin.left}
          y1={margin.top + plotHeight}
          x2={margin.left + plotWidth}
          y2={margin.top + plotHeight}
          stroke="#64748b"
        />
        {showReference && (
          <line
            x1={x(0)}
            y1={y(0)}
            x2={x(Math.min(1, yMax))}
            y2={y(Math.min(1, yMax))}
            stroke="#111827"
            strokeDasharray="4 4"
            strokeWidth={1.2}
          />
        )}
        {finitePoints.map((point, index) => (
          <circle
            key={`${point.row ?? "x"}-${point.col ?? "x"}-${point.action ?? "x"}-${index}`}
            cx={x(point.count_bonus)}
            cy={y(point.learned_bonus)}
            r={2.6}
            fill="#2563eb"
            fillOpacity={0.62}
          >
            <title>
              count {point.count}, count bonus {formatValue(point.count_bonus)}, learned {formatValue(point.learned_bonus)}
            </title>
          </circle>
        ))}
        <text x={margin.left + plotWidth / 2} y={height - 8} textAnchor="middle" fontSize={11} fill="#475569">
          count-based bonus
        </text>
        <text
          x={14}
          y={margin.top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${margin.top + plotHeight / 2})`}
          fontSize={11}
          fill="#475569"
        >
          {yLabel}
        </text>
      </svg>
    </section>
  );
}

function cellPolygons(row: number, col: number): string[] {
  const center = `${col + 0.5},${row + 0.5}`;
  return [
    `${col},${row} ${col + 1},${row} ${center}`,
    `${col},${row + 1} ${col + 1},${row + 1} ${center}`,
    `${col},${row} ${col},${row + 1} ${center}`,
    `${col + 1},${row} ${col + 1},${row + 1} ${center}`,
  ];
}

function flattenValues(values: ValueMatrix | number[][], validMask?: boolean[][]): number[] {
  return values.flatMap((row, rowIndex) =>
    row
      .filter((value, colIndex) => (validMask?.[rowIndex]?.[colIndex] ?? true) && value !== null)
      .map((value) => Number(value)),
  );
}

function flattenActionValues(values: ActionValueMatrix | number[][][], validMask?: boolean[][]): number[] {
  const flattened: number[] = [];
  values.forEach((row, rowIndex) => {
    row.forEach((actions, colIndex) => {
      if (!(validMask?.[rowIndex]?.[colIndex] ?? true)) return;
      actions.forEach((value) => {
        if (value !== null) flattened.push(Number(value));
      });
    });
  });
  return flattened;
}

function valueDomain(values: number[]): { min: number; max: number } {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (Math.abs(max - min) < 1e-9) return { min, max: min + 1 };
  return { min, max };
}

function colorForValue(value: number, domain: { min: number; max: number }, palette: "count" | "bonus"): string {
  const raw = (value - domain.min) / (domain.max - domain.min);
  const t = Math.max(0, Math.min(1, palette === "count" ? Math.sqrt(raw) : raw));
  const stops = VIRIDIS_STOPS;
  for (let idx = 1; idx < stops.length; idx += 1) {
    if (t <= stops[idx][0]) {
      const [leftT, leftColor] = stops[idx - 1];
      const [rightT, rightColor] = stops[idx];
      const localT = (t - leftT) / Math.max(rightT - leftT, 1e-9);
      return rgb(interpolateColor(leftColor, rightColor, localT));
    }
  }
  return rgb(stops[stops.length - 1][1]);
}

const VIRIDIS_STOPS: Array<[number, [number, number, number]]> = [
  [0, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1, [253, 231, 37]],
];

function interpolateColor(
  left: [number, number, number],
  right: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    Math.round(left[0] + (right[0] - left[0]) * t),
    Math.round(left[1] + (right[1] - left[1]) * t),
    Math.round(left[2] + (right[2] - left[2]) * t),
  ];
}

function rgb(color: [number, number, number]): string {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function textColor(fill: string): string {
  const match = fill.match(/\d+/g);
  if (!match || match.length < 3) return "#111827";
  const [red, green, blue] = match.slice(0, 3).map(Number);
  const luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  return luminance > 148 ? "#111827" : "#ffffff";
}

function formatValue(value: number): string {
  if (Math.abs(value) >= 100 || Number.isInteger(value)) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toPrecision(3);
}

function Legend({
  domain,
  palette,
  valueLabel,
}: {
  domain: { min: number; max: number };
  palette: "count" | "bonus";
  valueLabel: string;
}) {
  const gradientId = `${palette}-${valueLabel.replace(/\W+/g, "-")}`;
  return (
    <div className="heatmap-legend">
      <svg viewBox="0 0 120 10" aria-hidden="true">
        <defs>
          <linearGradient id={gradientId}>
            {VIRIDIS_STOPS.map(([offset, color]) => (
              <stop key={offset} offset={`${offset * 100}%`} stopColor={rgb(color)} />
            ))}
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="120" height="10" fill={`url(#${gradientId})`} rx="2" />
      </svg>
      <span>{formatValue(domain.min)}</span>
      <span>{formatValue(domain.max)}</span>
    </div>
  );
}

function PlotHeader({
  title,
  svgRef,
}: {
  title: string;
  svgRef: RefObject<SVGSVGElement>;
}) {
  return (
    <div className="plot-header">
      <div className="panel-title">{title}</div>
      <button
        type="button"
        className="icon-button"
        aria-label={`Export ${title}`}
        title={`Export ${title}`}
        onClick={() => exportSvg(svgRef.current, title)}
      >
        <Download size={15} />
      </button>
    </div>
  );
}

function exportSvg(svg: SVGSVGElement | null, title: string): void {
  if (!svg) return;
  const cloned = svg.cloneNode(true) as SVGSVGElement;
  cloned.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const source = new XMLSerializer().serializeToString(cloned);
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${slugifyFilename(title)}.svg`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function slugifyFilename(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return slug || "plot";
}
