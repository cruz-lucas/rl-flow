import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";

export function ExperimentDetail() {
  const experiments = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const latest = experiments.data?.[0];
  const experimentId = String(latest?.experiment_id ?? "");
  const artifacts = useQuery({
    queryKey: ["artifacts", experimentId],
    queryFn: () => api.artifacts(experimentId),
    enabled: Boolean(experimentId),
  });
  return (
    <main className="page">
      <header className="page-header">
        <h1>Experiment</h1>
      </header>
      {!latest ? (
        <div className="empty-state">No experiments</div>
      ) : (
        <div className="detail-grid">
          <section>
            <h2>{String(latest.experiment_id)}</h2>
            <p>{String(latest.run_dir)}</p>
            <code>{String(latest.command)}</code>
          </section>
          <section>
            <h2>Artifacts</h2>
            <ul>
              {(artifacts.data ?? []).map((path) => (
                <li key={path}>{path}</li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </main>
  );
}
