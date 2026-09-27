from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_assistant.main import build_retriever
from rag_assistant.prompting.prompt_builder import (
    generate_answer_or_shortcircuit,
    build_prompt,
    format_answer_with_citations,
)
from rag_assistant.generation.llm import generate_answer

app = FastAPI(title="ML Knowledge Assistant")

# Built once, at server startup — not per-request. Same reasoning as
# model-loading in embedder.py and llm.py.
retriever = build_retriever()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    """
    Simple liveness check — does the service respond at all.
    Deliberately does NOT touch the retriever, embedder, or LLM — a
    deployment system (or load balancer) calls this frequently to decide
    whether to route traffic here, and it needs to be near-instant and
    cheap, not a full pipeline run.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accept a question, run it through the existing pipeline, return the answer.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    results = retriever.retrieve(question, k=5)
    shortcircuit = generate_answer_or_shortcircuit(question, results)

    if shortcircuit is not None:
        return QueryResponse(answer=shortcircuit)

    prompt = build_prompt(question, results)
    answer = generate_answer(prompt, max_new_tokens=300, temperature=0.1)
    final_answer = format_answer_with_citations(answer, results)

    return QueryResponse(answer=final_answer)