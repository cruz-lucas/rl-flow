import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "../../api/client";

export function JobsPage() {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3000 });
  return (
    <main className="page">
      <header className="page-header">
        <h1>Jobs</h1>
        <button onClick={() => jobs.refetch()}>
          <RefreshCw size={16} />
          Refresh
        </button>
      </header>
      <table className="data-table">
        <thead>
          <tr>
            <th>Job</th>
            <th>Status</th>
            <th>Backend</th>
            <th>Experiment</th>
            <th>Run directory</th>
          </tr>
        </thead>
        <tbody>
          {(jobs.data ?? []).map((job) => (
            <tr key={job.job_id}>
              <td>{job.job_id}</td>
              <td>{job.status?.state ?? String((job as unknown as { status: string }).status)}</td>
              <td>{job.backend}</td>
              <td>{job.experiment_id}</td>
              <td>{job.run_dir}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
