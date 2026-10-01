# src/rag_assistant/job_runner.py
import uuid
import traceback
from threading import Thread

from rag_assistant.pipeline_builder import build_pipeline
from rag_assistant.pipeline_registry import EMBEDDING_MODELS, LLM_MODELS
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.evaluation.evaluator import evaluate

# In-memory job store: fine for a single-user, local dashboard.
_jobs: dict[str, dict] = {}


def start_job(pdf_path: str, eval_set: list[dict], config: dict) -> str:
    """
    Create a new job, run the pipeline + evaluation in a background thread,
    and return a job_id immediately (does not block).
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None, "error": None}

    def _run():
        try:
            # Translate the display names the UI sends into real Hugging Face IDs
            embedding_id = EMBEDDING_MODELS[config["embedding_model"]]
            llm_id = LLM_MODELS[config["llm_model"]] if config.get("llm_model") else None
            print(f"[job {job_id}] llm_model in config: {config.get('llm_model')!r} -> llm_id: {llm_id!r}")
            store = build_pipeline(
                pdf_path=pdf_path,
                loader_name=config["loader"],
                splitter_name=config["splitter"],
                chunker_name=config["chunker"],
                embedding_model_name=embedding_id,
                chunk_size=config.get("chunk_size", 500),
                chunk_overlap=config.get("chunk_overlap", 50),
            )
            retriever = Retriever(store, distance_threshold=config.get("distance_threshold", 1.0))
            result = evaluate(
                retriever,
                eval_set,
                k=config.get("k", 5),
                llm_model_name=llm_id,
            )

            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"

    Thread(target=_run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)