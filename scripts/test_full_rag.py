# scripts/test_full_rag.py
from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_documents
from rag_assistant.embedding.embedder import embed_documents
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.prompting.prompt_builder import generate_answer_or_shortcircuit, build_prompt
from rag_assistant.generation.llm import generate_answer


def main():
    docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
    headers = detect_repeating_headers(docs, min_frequency=0.05)
    cleaned_docs = clean_documents(docs, headers)
    chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)

    vectors = embed_documents([c.text for c in chunks])

    store = VectorStore(dimension=384)
    store.add(vectors, chunks)

    retriever = Retriever(store, distance_threshold=1.0)

    for query in ["What is a partial dependence plot?", "how to cook a perfect llm?"]:
        print(f"\n{'='*60}\nQuestion: {query}\n{'='*60}")

        results = retriever.retrieve(query, k=5)
        shortcircuit = generate_answer_or_shortcircuit(query, results)

        if shortcircuit is not None:
            print(f"[No LLM called]\n{shortcircuit}")
        else:
            prompt = build_prompt(query, results)
            answer = generate_answer(prompt, max_new_tokens=300, temperature=0.1)
            print(f"[LLM-generated answer]\n{answer}")


if __name__ == "__main__":
    main()