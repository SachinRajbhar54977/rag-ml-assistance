from rag_assistant.retrieval.retriever import RetrievalResult

def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    """
    Construct the full prompt to send to the LLM, given a user's
    question and the retrieved, threshold-passed context chunks.

    If `results` is empty, this function formats an explicit 'No context' block
    so the LLM receives clean instructions not to speculate, unless the caller
    chooses to short-circuit before invoking the LLM.
    """
    # 1. Format the retrieved context block
    if results:
        context_blocks = []
        for i, res in enumerate(results, start=1):
            source = getattr(res.chunk, "source", "Unknown")
            page = getattr(res.chunk, "page", "N/A")
            text = res.chunk.text.strip()
            
            block = f"[Source {i}: {source} (Page {page})]\n{text}"
            context_blocks.append(block)
            
        formatted_context = "\n\n".join(context_blocks)
    else:
        formatted_context = "No relevant context found in the knowledge base."

    # 2. Assemble the grounding prompt
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer the question, state clearly that you do not know or that the information is not available in the provided documents. Do not rely on outside knowledge or hallucinate details.do not include inline citation markers like [Source 1]

---
CONTEXT:
{formatted_context}
---

QUESTION:
{question.strip()}

ANSWER:"""

    return prompt



def generate_answer_or_shortcircuit(question: str, results: list[RetrievalResult]) -> str | None:
    """
    Returns a canned 'not found' message immediately if results is empty
    (no LLM call needed). Returns None if results is non-empty, signaling
    the caller should proceed to build a prompt and call the LLM.
    """
    if not results:
        return "I couldn't find any relevant information in the knowledge base to answer your question."
    
    return None



def format_answer_with_citations(answer: str, results: list[RetrievalResult]) -> str:
    """
    Append a 'Sources' section listing the unique source/page combinations
    from `results`, in the order they first appear.
    """
    if not results:
        return answer.strip()

    # Track unique (source, page) pairs preserving first appearance order
    seen_sources: set[tuple[str, str | int]] = set()
    unique_sources: list[tuple[str, str | int]] = []

    for res in results:
        source = getattr(res.chunk, "source", "Unknown Source")
        page = getattr(res.chunk, "page", "N/A")
        key = (source, page)

        if key not in seen_sources:
            seen_sources.add(key)
            unique_sources.append(key)

    if not unique_sources:
        return answer.strip()

    # Build the formatted sources block
    sources_text = "\n".join(
        f"- {source} (Page {page})" for source, page in unique_sources
    )

    return f"{answer.strip()}\n\nSources:\n{sources_text}"