"""Module for embedding documents and queries using SentenceTransformers."""

from sentence_transformers import SentenceTransformer

# Module-level cache for loaded SentenceTransformer models
_loaded_models: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached model instance for model_name, loading it from

    Hugging Face only the first time it's requested.
    """
    if model_name not in _loaded_models:
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


def embed_documents(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[list[float]]:
    """Generate dense vector embeddings for a list of document strings."""
    if not texts:
        return []

    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """Generate a dense vector embedding for a single search query string."""
    model = _get_model(model_name)
    embedding = model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embedding.tolist()