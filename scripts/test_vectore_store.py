from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_documents
from rag_assistant.embedding.embedder import embed_documents, embed_query
from rag_assistant.vectorstore.faiss_store import VectorStore

docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
headers = detect_repeating_headers(docs, min_frequency=0.05)
cleaned_docs = clean_documents(docs, headers)
chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)

vectors = embed_documents([c.text for c in chunks])

store = VectorStore(dimension=384)
store.add(vectors, chunks)

print(f"Index now contains {store.index.ntotal} vectors")

query = "How to cook a perfect llm?"
query_vector = embed_query(query)

results = store.search(query_vector, k=5)

for rank, (chunk, distance) in enumerate(results, start=1):
    print(f"\n--- Rank {rank} (distance={distance:.4f}, page={chunk.page}) ---")
    print(chunk.text[:200])