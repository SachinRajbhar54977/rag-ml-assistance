import time

from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_documents
from rag_assistant.embedding.embedder import embed_documents


def main():
    pdf_path = RAW_DATA_DIR / "machine_learning.pdf"

    docs = load_pdf(pdf_path)
    headers = detect_repeating_headers(docs, min_frequency=0.05)
    cleaned_docs = clean_documents(docs, headers)
    chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)

    print(f"Embedding {len(chunks)} chunks...")

    texts = [c.text for c in chunks]

    start = time.time()
    vectors = embed_documents(texts)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.2f} seconds")
    print(f"Number of vectors: {len(vectors)}")
    print(f"Dimensions per vector: {len(vectors[0])}")
    print(f"Sample of vector 0 (first 5 values): {vectors[0][:5]}")


if __name__ == "__main__":
    main()