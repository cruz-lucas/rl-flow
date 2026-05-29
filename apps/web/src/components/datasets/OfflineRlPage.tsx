import { useMutation, useQuery } from "@tanstack/react-query";
import { Brain, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ActionHeatmap, BonusScatter, StateHeatmap } from "./Heatmaps";

type Granularity = "state" | "state_action";
type OfflineAlgorithm = "rnd" | "cfn" | "classifier" | "simhash";
type Activation = "relu" | "tanh" | "gelu" | "elu" | "linear";
type Optimizer = "adam" | "sgd" | "rmsprop";
type ActionConditioning = "none" | "input" | "output" | "pair";
type SimHashMode = "static" | "learned";

export function OfflineRlPage() {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [path, setPath] = useState("");
  const [algorithm, setAlgorithm] = useState<OfflineAlgorithm>("rnd");
  const [granularity, setGranularity] = useState<Granularity>("state");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(128);
  const [learningRate, setLearningRate] = useState(0.001);
  const [hiddenUnits, setHiddenUnits] = useState("128,128");
  const [activation, setActivation] = useState<Activation>("relu");
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [actionConditioning, setActionConditioning] = useState<ActionConditioning>("none");
  const [updatePeriod, setUpdatePeriod] = useState(1);
  const [outputDim, setOutputDim] = useState(64);
  const [intrinsicRewardScale, setIntrinsicRewardScale] = useState(1);
  const [intrinsicStatsDecay, setIntrinsicStatsDecay] = useState(0.99);
  const [intrinsicRewardEpsilon, setIntrinsicRewardEpsilon] = useState(0.0001);
  const [intrinsicRewardClip, setIntrinsicRewardClip] = useState("10");
  const [intrinsicRewardCenter, setIntrinsicRewardCenter] = useState(false);
  const [maxGradNorm, setMaxGradNorm] = useState(1);
  const [seed, setSeed] = useState(0);
  const [simhashMode, setSimhashMode] = useState<SimHashMode>("static");
  const [simhashBits, setSimhashBits] = useState(32);
  const [simhashTableSize, setSimhashTableSize] = useState(16384);
  const [simhashBonusExponent, setSimhashBonusExponent] = useState(0.5);
  const [simhashMinCount, setSimhashMinCount] = useState(1);
  const analysis = useMutation({
    mutationFn: () =>
      api.trainOfflineRnd({
        path,
        algorithm,
        granularity,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        hidden_units: parseHiddenUnits(hiddenUnits),
        activation,
        optimizer,
        action_conditioning: actionConditioning,
        update_period: updatePeriod,
        output_dim: outputDim,
        intrinsic_reward_scale: intrinsicRewardScale,
        intrinsic_stats_decay: intrinsicStatsDecay,
        intrinsic_reward_epsilon: intrinsicRewardEpsilon,
        intrinsic_reward_clip: intrinsicRewardClip.trim() === "" ? null : Number(intrinsicRewardClip),
        intrinsic_reward_center: intrinsicRewardCenter,
        max_grad_norm: maxGradNorm,
        seed,
        simhash_mode: simhashMode,
        simhash_bits: simhashBits,
        simhash_table_size: simhashTableSize,
        simhash_bonus_exponent: simhashBonusExponent,
        simhash_min_count: simhashMinCount,
      }),
  });

  useEffect(() => {
    if (!path && datasets.data?.[0]) {
      setPath(datasets.data[0].path);
    }
  }, [datasets.data, path]);

  const result = analysis.data;
  const activeAlgorithm = result?.algorithm ?? algorithm;
  const algorithmLabel = algorithmLabels[activeAlgorithm] ?? activeAlgorithm;
  const learnedValueLabel = activeAlgorithm === "classifier" ? "Unknown Probability" : "learned bonus";
  const learnedPlotLabel = activeAlgorithm === "classifier" ? "Classifier Unknown Probability" : `${algorithmLabel} Bonus`;
  return (
    <main className="page offline-page">
      <div className="page-header">
        <h1>
          <Brain size={20} />
          Offline RL
        </h1>
        <button onClick={() => analysis.mutate()} disabled={analysis.isPending || path.trim().length === 0}>
          <Play size={16} />
          Run {algorithmLabels[algorithm]}
        </button>
      </div>
      <div className="offline-controls">
        <label className="field wide">
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
          <span>algorithm</span>
          <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as OfflineAlgorithm)}>
            <option value="rnd">RND</option>
            <option value="cfn">CFN</option>
            <option value="classifier">Known/Unknown Classifier</option>
            <option value="simhash">SimHash</option>
          </select>
        </label>
        <label className="field">
          <span>granularity</span>
          <select value={granularity} onChange={(event) => setGranularity(event.target.value as Granularity)}>
            <option value="state">State</option>
            <option value="state_action">State-Action</option>
          </select>
        </label>
        <label className="field">
          <span>epochs</span>
          <input
            type="number"
            min={1}
            max={2000}
            step={1}
            value={epochs}
            onChange={(event) => setEpochs(Number.parseInt(event.target.value, 10) || 1)}
          />
        </label>
        <label className="field">
          <span>batch size</span>
          <input
            type="number"
            min={1}
            max={8192}
            step={1}
            value={batchSize}
            onChange={(event) => setBatchSize(Number.parseInt(event.target.value, 10) || 1)}
          />
        </label>
        <label className="field">
          <span>learning rate</span>
          <input
            type="number"
            min={0.000001}
            max={1}
            step="any"
            value={learningRate}
            onChange={(event) => setLearningRate(Number(event.target.value) || 0.001)}
          />
        </label>
        <label className="field">
          <span>hidden units</span>
          <input value={hiddenUnits} onChange={(event) => setHiddenUnits(event.target.value)} />
        </label>
        <label className="field">
          <span>activation</span>
          <select value={activation} onChange={(event) => setActivation(event.target.value as Activation)}>
            <option value="relu">relu</option>
            <option value="tanh">tanh</option>
            <option value="gelu">gelu</option>
            <option value="elu">elu</option>
            <option value="linear">linear</option>
          </select>
        </label>
        <label className="field">
          <span>optimizer</span>
          <select value={optimizer} onChange={(event) => setOptimizer(event.target.value as Optimizer)}>
            <option value="adam">adam</option>
            <option value="sgd">sgd</option>
            <option value="rmsprop">rmsprop</option>
          </select>
        </label>
        <label className="field">
          <span>action conditioning</span>
          <select
            value={actionConditioning}
            onChange={(event) => setActionConditioning(event.target.value as ActionConditioning)}
          >
            <option value="none">none</option>
            <option value="input">input</option>
            <option value="output">output</option>
            <option value="pair">pair</option>
          </select>
        </label>
        <label className="field">
          <span>update period</span>
          <input
            type="number"
            min={1}
            step={1}
            value={updatePeriod}
            onChange={(event) => setUpdatePeriod(Number.parseInt(event.target.value, 10) || 1)}
          />
        </label>
        <label className="field">
          <span>output dim</span>
          <input
            type="number"
            min={1}
            max={4096}
            step={1}
            value={outputDim}
            onChange={(event) => setOutputDim(Number.parseInt(event.target.value, 10) || 1)}
          />
        </label>
        <label className="field">
          <span>reward scale</span>
          <input
            type="number"
            min={0}
            step="any"
            value={intrinsicRewardScale}
            onChange={(event) => setIntrinsicRewardScale(Number(event.target.value) || 0)}
          />
        </label>
        <label className="field">
          <span>stats decay</span>
          <input
            type="number"
            min={0}
            max={1}
            step="any"
            value={intrinsicStatsDecay}
            onChange={(event) => setIntrinsicStatsDecay(Number(event.target.value) || 0)}
          />
        </label>
        <label className="field">
          <span>reward epsilon</span>
          <input
            type="number"
            min={0.0000001}
            step="any"
            value={intrinsicRewardEpsilon}
            onChange={(event) => setIntrinsicRewardEpsilon(Number(event.target.value) || 0.0001)}
          />
        </label>
        <label className="field">
          <span>reward clip</span>
          <input value={intrinsicRewardClip} onChange={(event) => setIntrinsicRewardClip(event.target.value)} />
        </label>
        <label className="field checkbox-field">
          <span>center reward</span>
          <input
            type="checkbox"
            checked={intrinsicRewardCenter}
            onChange={(event) => setIntrinsicRewardCenter(event.target.checked)}
          />
        </label>
        <label className="field">
          <span>max grad norm</span>
          <input
            type="number"
            min={0}
            step="any"
            value={maxGradNorm}
            onChange={(event) => setMaxGradNorm(Number(event.target.value) || 0)}
          />
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
        {algorithm === "simhash" && (
          <>
            <label className="field">
              <span>simhash mode</span>
              <select value={simhashMode} onChange={(event) => setSimhashMode(event.target.value as SimHashMode)}>
                <option value="static">static</option>
                <option value="learned">learned</option>
              </select>
            </label>
            <label className="field">
              <span>simhash bits</span>
              <input
                type="number"
                min={1}
                max={4096}
                step={1}
                value={simhashBits}
                onChange={(event) => setSimhashBits(Number.parseInt(event.target.value, 10) || 1)}
              />
            </label>
            <label className="field">
              <span>hash table size</span>
              <input
                type="number"
                min={1}
                step={1}
                value={simhashTableSize}
                onChange={(event) => setSimhashTableSize(Number.parseInt(event.target.value, 10) || 1)}
              />
            </label>
            <label className="field">
              <span>bonus exponent</span>
              <input
                type="number"
                min={0.000001}
                step="any"
                value={simhashBonusExponent}
                onChange={(event) => setSimhashBonusExponent(Number(event.target.value) || 0.5)}
              />
            </label>
            <label className="field">
              <span>min count</span>
              <input
                type="number"
                min={0.000001}
                step="any"
                value={simhashMinCount}
                onChange={(event) => setSimhashMinCount(Number(event.target.value) || 1)}
              />
            </label>
          </>
        )}
      </div>
      {analysis.error && <div className="error-state">{analysis.error.message}</div>}
      {result && (
        <div className="offline-results">
          <section className="summary-strip">
            <div>
              <span>algorithm</span>
              <strong>{algorithmLabel}</strong>
            </div>
            <div>
              <span>unique</span>
              <strong>{result.unique_items}</strong>
            </div>
            <div>
              <span>epochs</span>
              <strong>{result.epochs}</strong>
            </div>
            <div>
              <span>final loss</span>
              <strong>{formatMetric(result.loss_history[result.loss_history.length - 1])}</strong>
            </div>
          </section>
          <div className="heatmap-grid">
            {result.granularity === "state_action" ? (
              <>
                {result.learned_state_action_bonus && result.visitation && (
                  <ActionHeatmap
                    title={`${learnedPlotLabel} by State-Action`}
                    values={result.learned_state_action_bonus}
                    validMask={result.visitation.valid_mask}
                    actionLabels={result.visitation.action_labels}
                    valueLabel={learnedValueLabel}
                    palette="bonus"
                  />
                )}
                {result.count_state_action_bonus && result.visitation && (
                  <ActionHeatmap
                    title="Count-Based State-Action Bonus"
                    values={result.count_state_action_bonus}
                    validMask={result.visitation.valid_mask}
                    actionLabels={result.visitation.action_labels}
                    valueLabel="count bonus"
                    palette="bonus"
                  />
                )}
              </>
            ) : (
              <>
                {result.learned_state_bonus && result.visitation && (
                  <StateHeatmap
                    title={`${learnedPlotLabel} by State`}
                    values={result.learned_state_bonus}
                    validMask={result.visitation.valid_mask}
                    valueLabel={learnedValueLabel}
                    palette="bonus"
                  />
                )}
                {result.count_state_bonus && result.visitation && (
                  <StateHeatmap
                    title="Count-Based State Bonus"
                    values={result.count_state_bonus}
                    validMask={result.visitation.valid_mask}
                    valueLabel="count bonus"
                    palette="bonus"
                  />
                )}
              </>
            )}
            <BonusScatter
              points={result.scatter}
              title={`${algorithmLabel} vs Count-Based Bonus`}
              yLabel={learnedValueLabel}
            />
          </div>
        </div>
      )}
    </main>
  );
}

const algorithmLabels: Record<string, string> = {
  rnd: "RND",
  cfn: "CFN",
  classifier: "Known/Unknown Classifier",
  simhash: "SimHash",
};

function formatMetric(value: number | undefined): string {
  if (value === undefined) return "";
  if (Math.abs(value) >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}

function parseHiddenUnits(value: string): number[] {
  const parsed = value
    .split(/[,\s]+/)
    .map((item) => Number.parseInt(item, 10))
    .filter((item) => Number.isFinite(item) && item > 0);
  return parsed.length > 0 ? parsed : [];
}
