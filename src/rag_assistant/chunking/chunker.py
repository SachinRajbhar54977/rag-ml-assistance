from dataclasses import dataclass

from rag_assistant.ingestion.pdf_loader import Document
import re

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a single Document's text into overlapping fixed-size chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.text

    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=document.source,
                    page=document.page,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk every document in the list, returning one flat list of all chunks."""
    all_chunks = []

    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    return all_chunks

# Add to src/rag_assistant/chunking/chunker.py, alongside Chunk and chunk_document




def split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: split on '. ', '! ', '? ' followed by a
    capital letter. Not perfect (mishandles abbreviations like "Dr." or
    "e.g."), but good enough as a first pass.
    """
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _fallback_char_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Last-resort splitter for a single sentence longer than chunk_size."""
    sub_chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            sub_chunks.append(piece)
        if end == len(text):
            break
        start += step
    return sub_chunks


def _split_paragraph_into_units(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Break one paragraph into sentence-sized (or smaller) text units."""
    if len(paragraph) <= chunk_size:
        return [paragraph]

    units = []
    for sentence in split_into_sentences(paragraph):
        if len(sentence) > chunk_size:
            units.extend(_fallback_char_chunks(sentence, chunk_size, chunk_overlap))
        else:
            units.append(sentence)
    return units


def chunk_document_recursive(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a Document's text by paragraph, then sentence, falling back
    to character-based splitting only when a single sentence exceeds
    chunk_size on its own. Units are packed greedily up to chunk_size,
    with trailing units carried forward as overlap into the next chunk.
    """
    if not document.text.strip():
        return []

    paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        units.extend(_split_paragraph_into_units(para, chunk_size, chunk_overlap))

    chunks: list[Chunk] = []
    current_units: list[str] = []
    current_len = 0
    chunk_index = 0

    def seal_chunk():
        nonlocal chunk_index
        text = " ".join(current_units).strip()
        if text:
            chunks.append(Chunk(
                text=text,
                source=document.source,
                page=document.page,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    for unit in units:
        candidate_len = current_len + len(unit) + (1 if current_units else 0)

        if current_units and candidate_len > chunk_size:
            seal_chunk()

            # Carry trailing units forward as overlap, up to chunk_overlap chars
            overlap_units = []
            overlap_len = 0
            for u in reversed(current_units):
                if overlap_len + len(u) > chunk_overlap:
                    break
                overlap_units.insert(0, u)
                overlap_len += len(u) + 1

            current_units = overlap_units
            current_len = sum(len(u) for u in current_units) + max(0, len(current_units) - 1)

        current_units.append(unit)
        current_len += len(unit) + (1 if len(current_units) > 1 else 0)

    seal_chunk()

    return chunks
