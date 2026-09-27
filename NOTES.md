# RAG ML Knowledge Assistant — Design Decisions & Known Limitations

This note documents the key engineering decisions made while building this project, the evidence behind them, and the limitations that remain — written the way a real handoff note should read, so a future maintainer (or future me) doesn't have to rediscover any of this from scratch.

## Pipeline summary

```
PDF (machine_learning.pdf, 251 pages)
  -> load_pdf()            PyMuPDF extraction
  -> clean_documents()     strip repeating headers / page numbers
  -> chunk_document_recursive()   paragraph -> sentence -> char fallback
  -> embed_documents()     all-MiniLM-L6-v2, 384-dim, GPU (CUDA)
  -> VectorStore           FAISS IndexFlatL2, persisted to disk
  -> Retriever             distance_threshold = 1.0, top-k = 5
  -> build_prompt()        context + explicit "don't hallucinate" instruction
  -> generate_answer()     Qwen2.5-1.5B-Instruct, fp16, temperature 0.1
  -> format_answer_with_citations()
```

## Key decisions and why

**PDF extraction: PyMuPDF, not pypdf/pdfplumber.**
pypdf and pdfplumber both produced systematically glued words (no spaces between words) on 100% of pages. A head-to-head comparison on a known-bad page showed PyMuPDF alone extracted correctly spaced text, with headers cleanly separated from body text. Fixing extraction at the source eliminated the glued-word problem entirely (250/250 pages affected -> 205/250 pages with zero long-token artifacts, remainder explained by legitimate long tokens: URLs, citations, footnote markers).

**Header/footer removal: frequency-based detection, not hardcoded strings.**
Running headers ("Interpretable Models", "Model-Agnostic Methods", etc.) were detected by finding lines whose first-line text repeats across a threshold fraction of pages (min_frequency=0.05, tuned empirically — lower thresholds found no new headers past this point, confirming it as the natural stopping point). Stripping is restricted to the first/last 2 lines of a page only, to avoid deleting legitimate numeric data (verified against a page containing a data table with bare-digit values in the middle of the page).

**Chunking: recursive (paragraph -> sentence -> char fallback), not fixed-size.**
Fixed-size character chunking cut chunks mid-word/mid-sentence, and was directly implicated in a real retrieval failure (a RuleFit definition chunk was never surfaced for a heavily rephrased query). Switching to recursive chunking fixed a separate confirmed false positive (an XGBoost question incorrectly matched content on tree ensembles) and produced cleaner, more complete chunk text. It did not fully resolve the RuleFit case — see Known limitation 1 below.

**Distance threshold: 1.0, on L2 distance over normalized embeddings.**
all-MiniLM-L6-v2 outputs unit-normalized vectors by default (verified directly, norms ≈ 1.0), so L2 distance ranking matches cosine similarity ranking without extra normalization code. Threshold was calibrated against two extremes (a clearly relevant query at 0.47–0.64, a nonsense query at 1.53–1.57) and confirmed against a 10-question eval set. Tested thresholds from 0.90–1.15: raising the threshold consistently reintroduced false positives (backprop/CNN and XGBoost questions matching irrelevant content) without ever recovering the RuleFit rephrase failure, since that case's best match sits at distance ~1.20 — beyond any threshold that doesn't also let in known-bad matches. 1.0 is the best available setting given the current embedding model.

**Generation model: Qwen2.5-1.5B-Instruct, fp16.**
Chosen for hardware fit (GTX 1650, 4GB VRAM, 8GB RAM) — fits comfortably in fp16 (~3GB) with headroom, verified via nvidia-smi during generation (39% GPU-util, 3065MB used). A quantized 3B+ model is a plausible future upgrade once there's a reason to believe answer quality is the bottleneck rather than retrieval.

## Known limitations

**1. Heavily rephrased queries can fail to retrieve the correct chunk, even when it exists in the index.**
Confirmed case: "If I want to turn a complex random forest model into a human-readable set of IF-THEN conditions using regularized regression, what algorithm should I run?" (a paraphrase of "How does RuleFit work?") never surfaces the RuleFit definition chunk (page 104) within the top 10 raw FAISS results at a competitive distance, regardless of chunking strategy. This is an embedding model capability limit, not a retrieval-logic bug. Candidate fixes, not yet implemented: a stronger/larger embedding model (e.g. a bge- or e5-family model), query expansion/rewriting before embedding, or a hybrid keyword+semantic search fallback.

**2. Fixed distance threshold is a single global cutoff, not adaptive.**
The same threshold is applied to every query regardless of question type or phrasing style. A rephrased query and a direct, book-vocabulary query do not necessarily produce comparable distance distributions. Not yet addressed.

**3. No reranking step.**
Retrieval returns FAISS's raw top-k by distance only. A cross-encoder reranking pass over a larger initial candidate set (e.g. retrieve k=15, rerank down to 5) is a likely high-value next step, particularly for the rephrasing failure above.

**4. Citations list every retrieved-and-passed-threshold chunk, not verified "actually used by the model" chunks.**
The LLM is not asked to confirm which sources it drew on; the Sources list is a deterministic dump of everything in context. This is a reasonable approximation but not a guarantee of per-sentence attribution.

**5. Single document, single-process, no concurrency or persistence beyond one FAISS index file.**
Adding a second document requires re-running the full ingestion pipeline from scratch; there's no incremental indexing. Not a concern at current scale, but a real limitation before any multi-document or multi-user use case.

## Evaluation

10-question hand-built eval set (`data/eval/eval_set.json`), covering definition, factual, conceptual-comparison, multi-chunk, rephrased-conceptual, and out-of-scope categories. Current measured accuracy: **7/10 (70%)**, with both failures traced to limitation #1 above — not hidden, not guessed at.

Ground-truth `expected_page` values in the eval set were independently verified against the actual cleaned document text after an initial draft (built from memory) produced several incorrect expectations — a reminder that eval-set quality has to be checked with the same rigor as the system under test.