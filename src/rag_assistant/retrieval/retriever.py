from dataclasses import dataclass
from rag_assistant.chunking.chunker import Chunk
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.embedding.embedder import embed_query

@dataclass
class RetrievalResult:
    chunk: Chunk
    distance: float

class Retriever:
    def __init__(self, store: VectorStore, distance_threshold: float = 1.0):
        """
        Wrap a VectorStore with a relevance threshold.
        """
        self.store = store
        self.distance_threshold = distance_threshold

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """
        Embed the query, search the store, and return only results
        whose distance is below self.distance_threshold.
        If no results pass the threshold, return an empty list —
        this is the signal for "not found in the knowledge base."
        """
        if not query.strip():
            return []

        # Convert text query into vector embedding
        query_vector = embed_query(query)

        # Search the vector store for top k candidates
        raw_results = self.store.search(query_vector, k=k)

        # Filter by distance threshold and wrap in RetrievalResult objects
        filtered_results = [
            RetrievalResult(chunk=chunk, distance=dist)
            for chunk, dist in raw_results
            if dist < self.distance_threshold
        ]

        return filtered_results