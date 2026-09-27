from rag_assistant.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_document_recursive
from rag_assistant.embedding.embedder import embed_documents
from rag_assistant.vectorstore.faiss_store import VectorStore


def main():
    docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
    headers = detect_repeating_headers(docs, min_frequency=0.05)
    cleaned_docs = clean_documents(docs, headers)

    chunks = []
    for doc in cleaned_docs:
        chunks.extend(chunk_document_recursive(doc, chunk_size=500, chunk_overlap=50))

    print(f"Recursive chunking produced {len(chunks)} chunks (vs 1222 with fixed-size)")

    vectors = embed_documents([c.text for c in chunks])

    store = VectorStore(dimension=384)
    store.add(vectors, chunks)

    index_path = PROCESSED_DATA_DIR / "faiss_index_recursive.bin"
    metadata_path = PROCESSED_DATA_DIR / "metadata_recursive.pkl"
    store.save(str(index_path), str(metadata_path))

    print(f"Saved recursive index to {index_path}")


if __name__ == "__main__":
    main()