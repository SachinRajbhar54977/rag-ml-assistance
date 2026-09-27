# src/rag_assistant/vectorstore/faiss_store.py
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag_assistant.chunking.chunker import Chunk


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: list[Chunk] = []

    def add(self, vectors: list[list[float]], chunks: list[Chunk]) -> None:
            """
            Add vectors to the FAISS index, and store their matching
            Chunk objects in the metadata list, at the same positions.
            """
            if len(vectors) != len(chunks):
                raise ValueError("The number of vectors and chunks must match to maintain synchronization.")

            if not vectors:
                return

            np_vectors = np.array(vectors, dtype=np.float32)

            self.index.add(np_vectors)
            self.metadata.extend(chunks)

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Search for the k nearest vectors to query_vector.
        Return a list of (Chunk, distance) pairs.
        """
        if self.index.ntotal == 0:
            return []

        np_query = np.array([query_vector], dtype=np.float32)

        distances, indices = self.index.search(np_query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.metadata[idx]
            results.append((chunk, float(dist)))

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata list to disk.
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load a previously saved FAISS index and metadata list from disk,
        replacing this instance's current index/metadata.
        """
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Synchronization mismatch! Index has {self.index.ntotal} vectors "
                f"but metadata store has {len(self.metadata)} items."
            )