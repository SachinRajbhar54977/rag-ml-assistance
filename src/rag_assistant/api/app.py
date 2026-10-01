# src/rag_assistant/api/app.py
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pathlib import Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.generation.llm import generate_answer
from rag_assistant.job_runner import get_job, start_job
from rag_assistant.main import build_retriever
from rag_assistant.pipeline_registry import (
    CHUNKERS,
    EMBEDDING_MODELS,
    LLM_MODELS,
    PDF_LOADERS,
    TEXT_SPLITTERS,
)
from rag_assistant.prompting.prompt_builder import (
    build_prompt,
    format_answer_with_citations,
    generate_answer_or_shortcircuit,
)

UPLOAD_DIR = RAW_DATA_DIR / "uploads"

app = FastAPI(title="ML Knowledge Assistant")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = build_retriever()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class RunResponse(BaseModel):
    job_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def get_options():
    """Everything the UI can offer in its dropdowns, straight from the registries."""
    return {
        "loaders": list(PDF_LOADERS.keys()),
        "splitters": list(TEXT_SPLITTERS.keys()),
        "chunkers": list(CHUNKERS.keys()),
        "embedding_models": list(EMBEDDING_MODELS.keys()),
        "llm_models": list(LLM_MODELS.keys()),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    results = retriever.retrieve(question, k=5)
    shortcircuit = generate_answer_or_shortcircuit(question, results)

    if shortcircuit is not None:
        return QueryResponse(answer=shortcircuit)

    prompt = build_prompt(question, results)
    answer = generate_answer(prompt, max_new_tokens=300, temperature=0.1)
    return QueryResponse(answer=format_answer_with_citations(answer, results))


@app.post("/runs", response_model=RunResponse)
async def create_run(
    pdf_file: UploadFile = File(...),
    eval_file: UploadFile = File(...),
    loader: str = Form(...),
    splitter: str = Form(...),
    chunker: str = Form(...),
    embedding_model: str = Form(...),
    llm_model: str | None = Form(None),  # leave blank for retrieval-only
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    distance_threshold: float = Form(1.0),
    k: int = Form(5),
):
    """
    Accept a PDF and an eval-set JSON file, plus pipeline configuration as
    form fields, and start a background job.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Path(...).name strips any directory parts from the client-supplied filename
    pdf_path = UPLOAD_DIR / Path(pdf_file.filename).name
    pdf_path.write_bytes(await pdf_file.read())

    try:
        eval_set = json.loads(await eval_file.read())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"eval_file is not valid JSON: {e}")

    config = {
        "loader": loader,
        "splitter": splitter,
        "chunker": chunker,
        "embedding_model": embedding_model,
        "llm_model": llm_model or None,  # a blank form field arrives as "", so map it to None
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "distance_threshold": distance_threshold,
        "k": k,
    }
    print(f"[create_run] config = {config}")

    job_id = start_job(str(pdf_path), eval_set, config)
    return RunResponse(job_id=job_id)


@app.get("/runs/{job_id}")
def get_run(job_id: str):
    """Return the current status (and result, if done) of a job."""
    job = get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found with id {job_id!r}")

    return job