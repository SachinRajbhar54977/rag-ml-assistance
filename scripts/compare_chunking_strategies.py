from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.chunking.chunker import chunk_documents, chunk_document_recursive
from rag_assistant.embedding.embedder import embed_documents, embed_query
from rag_assistant.vectorstore.faiss_store import VectorStore

docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
headers = detect_repeating_headers(docs, min_frequency=0.05)
cleaned_docs = clean_documents(docs, headers)

# Build chunks with the NEW recursive strategy across all documents
recursive_chunks = []
for doc in cleaned_docs:
    recursive_chunks.extend(chunk_document_recursive(doc, chunk_size=500, chunk_overlap=50))

print(f"Fixed-size chunking would produce a different count; recursive produced {len(recursive_chunks)} chunks")

vectors = embed_documents([c.text for c in recursive_chunks])

store = VectorStore(dimension=384)
store.add(vectors, recursive_chunks)

query = "If I want to turn a complex random forest model into a human-readable set of IF-THEN conditions using regularized regression, what algorithm should I run?"
query_vector = embed_query(query)

raw_results = store.search(query_vector, k=10)

for rank, (chunk, distance) in enumerate(raw_results, start=1):
    print(f"Rank {rank}: page={chunk.page}, distance={distance:.4f}")
    print(f"  {chunk.text[:100]!r}")