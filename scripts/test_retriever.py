# scripts/test_retriever.py
from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_documents
from rag_assistant.embedding.embedder import embed_documents
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever


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
        print(f"\n=== Query: {query!r} ===")
        results = retriever.retrieve(query, k=5)
        print(f"{len(results)} result(s) passed the threshold")
        for r in results:
            print(f"  page={r.chunk.page}, distance={r.distance:.4f}")


if __name__ == "__main__":
    main()