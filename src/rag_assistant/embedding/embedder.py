# src/rag_assistant/embedding/embedder.py
from sentence_transformers import SentenceTransformer

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device sentence-transformers will use:", "cuda" if torch.cuda.is_available() else "cpu")

_MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once, at import time — not inside the functions below.
# Loading involves reading model weights from disk and initializing the
# network; doing that on every call (e.g. once per chunk, 1222 times)
# would be extremely slow. One shared instance is reused for every call.
model = SentenceTransformer(_MODEL_NAME)


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a batch of document chunk texts.
    Returns one embedding vector (as a plain list of floats) per input text,
    in the same order.

    batch_size controls how many texts are processed through the model at
    once — keeps memory usage bounded on limited-VRAM GPUs, regardless of
    how large `texts` is overall. sentence-transformers internally splits
    the full input into batches of this size and processes them sequentially.
    """
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single user query string.
    Returns one embedding vector as a plain list of floats.

    Kept as a separate function from embed_documents even though the
    underlying model call is currently identical. Some embedding models
    (not this one, but e.g. bge-*/e5-* families) expect queries and
    documents to be embedded differently — e.g. prefixed with "query: "
    vs "passage: " — because they were trained asymmetrically for better
    retrieval accuracy. Keeping these as two entry points means switching
    to such a model later only requires changing this function's internals,
    not every call site across the codebase.
    """
    embedding = model.encode(text)
    return embedding.tolist()