export interface OptionsResponse {
  loaders: string[];
  splitters: string[];
  chunkers: string[];
  embedding_models: string[];
  llm_models: string[];
}

export interface EvalDetail {
  question: string;
  category: string;
  should_find_answer: boolean;
  expected_page: number | null;
  retrieved_pages: number[];
  passed: boolean;
  generated_answer: string | null;
}

export interface EvalResult {
  accuracy: number;
  total: number;
  passed: number;
  categories: Record<string, { total: number; passed: number }>;
  details: EvalDetail[];
}

export interface JobStatus {
  status: "running" | "done" | "failed";
  result: EvalResult | null;
  error: string | null;
}