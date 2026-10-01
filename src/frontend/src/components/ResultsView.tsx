import type { JobStatus } from "../types";

interface Props {
  job: JobStatus | null;
}

export default function ResultsView({ job }: Props) {
  if (!job) return null;

  if (job.status === "running") {
    return <p className="status running">Running... this can take a few minutes if an LLM is selected.</p>;
  }

  if (job.status === "failed") {
    return (
      <div className="status failed">
        <h3>Run failed</h3>
        <pre>{job.error}</pre>
      </div>
    );
  }

  const result = job.result!;

  return (
    <div className="results">
      <h2>Accuracy: {(result.accuracy * 100).toFixed(1)}% ({result.passed}/{result.total})</h2>

      <table>
        <thead>
          <tr><th>Category</th><th>Passed</th></tr>
        </thead>
        <tbody>
          {Object.entries(result.categories).map(([cat, stats]) => (
            <tr key={cat}>
              <td>{cat}</td>
              <td>{stats.passed}/{stats.total}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Per-question detail</h3>
      {result.details.map((item, i) => (
        <div key={i} className={`detail-card ${item.passed ? "pass" : "fail"}`}>
          <p><strong>{item.question}</strong></p>
          <p>Category: {item.category} | Expected page: {item.expected_page ?? "—"} | Retrieved: {item.retrieved_pages.join(", ") || "none"}</p>
          {item.generated_answer && (
            <details>
              <summary>Generated answer</summary>
              <p>{item.generated_answer}</p>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}