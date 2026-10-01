from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents
from rag_assistant.embedding.embedder import embed_documents
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.pipeline_registry import PDF_LOADERS, TEXT_SPLITTERS, CHUNKERS


def build_pipeline(
    pdf_path: str,
    loader_name: str,
    splitter_name: str,
    chunker_name: str,
    embedding_model_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> VectorStore:
    """
    Build a complete VectorStore from a raw PDF using the chosen
    combination of pipeline components.
    """
    # Step A: look up and call the chosen loader
    loader = PDF_LOADERS[loader_name]
    docs = loader(pdf_path)

    # Step B: clean (not configurable yet — always the same)
    headers = detect_repeating_headers(docs, min_frequency=0.05)
    cleaned_docs = clean_documents(docs, headers)

    # Step C: look up the chosen splitter and chunker
    splitter = TEXT_SPLITTERS[splitter_name]
    chunker = CHUNKERS[chunker_name]

    # Step D: for every document, split into units, then pack into chunks
    all_chunks = []
    for doc in cleaned_docs:
        units = splitter(doc.text, chunk_size)
        chunks = chunker(units, source=doc.source, page=doc.page,
                          chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    # Step E: embed all chunks using the chosen embedding model
    vectors = embed_documents([c.text for c in all_chunks], model_name=embedding_model_name)

    # Step F: build and return the vector store
    store = VectorStore(dimension=len(vectors[0]))
    store.add(vectors, all_chunks)

    return store