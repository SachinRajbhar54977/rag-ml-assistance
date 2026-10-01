import axios from "axios";
import type { OptionsResponse, JobStatus } from "./types";

const BASE_URL = "http://localhost:8000";

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await axios.get<OptionsResponse>(`${BASE_URL}/options`);
  return res.data;
}

export interface RunConfig {
  pdfFile: File;
  evalFile: File;
  loader: string;
  splitter: string;
  chunker: string;
  embeddingModel: string;
  llmModel: string; // empty string means "retrieval only, no generation"
  chunkSize: number;
  chunkOverlap: number;
  distanceThreshold: number;
  k: number;
}

export async function startRun(config: RunConfig): Promise<string> {
  const form = new FormData();
  form.append("pdf_file", config.pdfFile);
  form.append("eval_file", config.evalFile);
  form.append("loader", config.loader);
  form.append("splitter", config.splitter);
  form.append("chunker", config.chunker);
  form.append("embedding_model", config.embeddingModel);
  if (config.llmModel) {
    form.append("llm_model", config.llmModel);
  }
  form.append("chunk_size", String(config.chunkSize));
  form.append("chunk_overlap", String(config.chunkOverlap));
  form.append("distance_threshold", String(config.distanceThreshold));
  form.append("k", String(config.k));

  const res = await axios.post<{ job_id: string }>(`${BASE_URL}/runs`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data.job_id;
}

export async function fetchJob(jobId: string): Promise<JobStatus> {
  const res = await axios.get<JobStatus>(`${BASE_URL}/runs/${jobId}`);
  return res.data;
}