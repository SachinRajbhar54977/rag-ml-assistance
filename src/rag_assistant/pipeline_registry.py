from rag_assistant.ingestion.pdf_loader import load_pdf  # your pymupdf-based loader
from rag_assistant.chunking.chunker import split_text_paragraph_sentence, split_text_raw, pack_units_fixed_size

PDF_LOADERS = {
    "pymupdf": load_pdf,
    # "pypdf": ...  # add once you've extracted your original pypdf version into its own named function
}

TEXT_SPLITTERS = {
    "paragraph_sentence": split_text_paragraph_sentence,
    "raw": split_text_raw,
}

CHUNKERS = {
    "fixed_size_char": pack_units_fixed_size,
}

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}

LLM_MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}
