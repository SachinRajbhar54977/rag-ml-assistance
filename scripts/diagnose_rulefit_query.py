from rag_assistant.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.embedding.embedder import embed_query

store = VectorStore(dimension=384)
store.load(
    str(PROCESSED_DATA_DIR / "faiss_index.bin"),
    str(PROCESSED_DATA_DIR / "metadata.pkl"),
)

query = "If I want to turn a complex random forest model into a human-readable set of IF-THEN conditions using regularized regression, what algorithm should I run?"
query_vector = embed_query(query)

# Bypass the Retriever entirely — call the raw store directly, no threshold, bigger k
raw_results = store.search(query_vector, k=10)

for rank, (chunk, distance) in enumerate(raw_results, start=1):
    print(f"Rank {rank}: page={chunk.page}, distance={distance:.4f}")
    print(f"  {chunk.text[:100]!r}")