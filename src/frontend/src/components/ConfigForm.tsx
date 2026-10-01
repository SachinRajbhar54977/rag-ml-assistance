import { useState, type FormEvent } from "react";
import type { OptionsResponse } from "../types";
import type { RunConfig } from "../api";

interface Props {
  options: OptionsResponse;
  onSubmit: (config: RunConfig) => void;
  disabled: boolean;
}

export default function ConfigForm({ options, onSubmit, disabled }: Props) {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [evalFile, setEvalFile] = useState<File | null>(null);
  const [loader, setLoader] = useState(options.loaders[0]);
  const [splitter, setSplitter] = useState(options.splitters[0]);
  const [chunker, setChunker] = useState(options.chunkers[0]);
  const [embeddingModel, setEmbeddingModel] = useState(options.embedding_models[0]);
  const [llmModel, setLlmModel] = useState(""); // "" = retrieval-only
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [distanceThreshold, setDistanceThreshold] = useState(1.0);
  const [k, setK] = useState(5);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();

    if (!pdfFile || !evalFile) {
      alert("Please choose both a PDF file and an eval-set JSON file.");
      return;
    }

    onSubmit({
      pdfFile,
      evalFile,
      loader,
      splitter,
      chunker,
      embeddingModel,
      llmModel,
      chunkSize,
      chunkOverlap,
      distanceThreshold,
      k,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="config-form">
      <h2>Configure a run</h2>

      <label>
        PDF file
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setPdfFile(e.target.files?.[0] ?? null)}
        />
      </label>

      <label>
        Eval set (JSON)
        <input
          type="file"
          accept="application/json"
          onChange={(e) => setEvalFile(e.target.files?.[0] ?? null)}
        />
      </label>

      <div className="grid">
        <label>
          PDF loader
          <select value={loader} onChange={(e) => setLoader(e.target.value)}>
            {options.loaders.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label>
          Text splitter
          <select value={splitter} onChange={(e) => setSplitter(e.target.value)}>
            {options.splitters.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label>
          Chunker
          <select value={chunker} onChange={(e) => setChunker(e.target.value)}>
            {options.chunkers.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label>
          Embedding model
          <select value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)}>
            {options.embedding_models.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>

        <label>
          LLM (optional — leave blank for retrieval-only)
          <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
            <option value="">(none — retrieval only)</option>
            {options.llm_models.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </label>
      </div>

      <details>
        <summary>Advanced parameters</summary>
        <div className="grid">
          <label>
            Chunk size
            <input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
          </label>
          <label>
            Chunk overlap
            <input type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} />
          </label>
          <label>
            Distance threshold
            <input type="number" step="0.05" value={distanceThreshold} onChange={(e) => setDistanceThreshold(Number(e.target.value))} />
          </label>
          <label>
            Top-k
            <input type="number" value={k} onChange={(e) => setK(Number(e.target.value))} />
          </label>
        </div>
      </details>

      <button type="submit" disabled={disabled}>
        {disabled ? "Run in progress..." : "Run pipeline"}
      </button>
    </form>
  );
}