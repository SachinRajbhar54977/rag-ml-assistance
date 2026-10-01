import { useEffect, useRef, useState } from "react";
import ConfigForm from "./components/ConfigForm";
import ResultsView from "./components/ResultsView";
import { fetchOptions, startRun, fetchJob } from "./api";
import type { OptionsResponse, JobStatus } from "./types";
import type { RunConfig } from "./api";
import "./App.css";

function App() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    fetchOptions()
      .then(setOptions)
      .catch(() => setError("Could not reach the backend. Is uvicorn running on port 8000?"));
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function handleSubmit(config: RunConfig) {
    setError(null);
    setJob({ status: "running", result: null, error: null });

    try {
      const jobId = await startRun(config);

      pollRef.current = window.setInterval(async () => {
        const status = await fetchJob(jobId);
        setJob(status);

        if (status.status === "done" || status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      }, 5000);
    } catch (e) {
      setError("Failed to start the run. Check the backend terminal for errors.");
      setJob(null);
    }
  }

  return (
    <div className="app">
      <h1>RAG Experimentation Dashboard</h1>

      {error && <p className="status failed">{error}</p>}

      {!options && !error && <p>Loading options from backend...</p>}

      {options && (
        <ConfigForm
          options={options}
          onSubmit={handleSubmit}
          disabled={job?.status === "running"}
        />
      )}

      <ResultsView job={job} />
    </div>
  );
}

export default App;