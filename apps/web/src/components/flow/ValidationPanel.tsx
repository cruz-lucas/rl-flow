import { Check, Play, Send, Terminal, Upload } from "lucide-react";
import type { ExperimentSpec, ValidationResult } from "../../types/schema";
import { useFlowStore } from "../../stores/flowStore";

interface ValidationPanelProps {
  validation?: ValidationResult;
  experiment?: ExperimentSpec;
  isBusy: boolean;
  onValidate: () => void;
  onCompile: () => void;
  onRun: (backend: "local" | "slurm") => void;
}

export function ValidationPanel({
  validation,
  experiment,
  isBusy,
  onValidate,
  onCompile,
  onRun,
}: ValidationPanelProps) {
  const { backend, setBackend } = useFlowStore();
  return (
    <footer className="bottom-panel">
      <div className="action-row">
        <button title="Validate" onClick={onValidate} disabled={isBusy}>
          <Check size={16} />
          Validate
        </button>
        <button title="Compile" onClick={onCompile} disabled={isBusy}>
          <Terminal size={16} />
          Compile
        </button>
        <div className="segmented" aria-label="Execution backend">
          <button className={backend === "local" ? "active" : ""} onClick={() => setBackend("local")}>
            local
          </button>
          <button className={backend === "slurm" ? "active" : ""} onClick={() => setBackend("slurm")}>
            slurm
          </button>
        </div>
        <button title="Run locally" onClick={() => onRun("local")} disabled={isBusy}>
          <Play size={16} />
          Run
        </button>
        <button title="Submit to SLURM" onClick={() => onRun("slurm")} disabled={isBusy}>
          <Upload size={16} />
          Submit
        </button>
      </div>
      <div className="feedback">
        {validation?.valid && <span className="ok">Workflow valid</span>}
        {validation && !validation.valid && (
          <ul>
            {validation.errors.map((error, index) => (
              <li key={`${error.code}-${index}`}>
                {error.node_id ? `${error.node_id}: ` : ""}
                {error.message}
              </li>
            ))}
          </ul>
        )}
        {experiment && (
          <code>
            {experiment.command}
          </code>
        )}
      </div>
    </footer>
  );
}
