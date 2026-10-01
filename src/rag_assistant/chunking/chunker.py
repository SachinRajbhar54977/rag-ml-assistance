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

# src/rag_assistant/chunking/chunker.py — replace the recursive-chunking section with this

def split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: split on '. ', '! ', '? ' followed by a
    capital letter or end of string.
    """
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _fallback_char_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fallback utility to slice a single text block larger than chunk_size into character chunks."""
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


def split_text_paragraph_sentence(text: str, chunk_size: int) -> list[str]:
    """
    Break text into paragraph/sentence-sized units.
    Paragraphs that fit within chunk_size stay whole; longer paragraphs
    are broken down into sentences.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(split_into_sentences(para))

    return units


def split_text_raw(text: str, chunk_size: int) -> list[str]:
    """
    No real splitting — return the whole text as a single unit.
    Lets the packer's char-fallback logic do all the work, matching
    the behavior of the original fixed-character chunker.
    """
    return [text] if text.strip() else []


def pack_units_fixed_size(
    units: list[str],
    source: str,
    page: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Greedy packing loop taking pre-split units as input.
    Packs units together into Chunks up to chunk_size, carrying trailing
    units forward as overlap. Falls back to character slicing if any
    single unit already exceeds chunk_size.
    """
    expanded_units = []
    for unit in units:
        if len(unit) > chunk_size:
            expanded_units.extend(_fallback_char_chunks(unit, chunk_size, chunk_overlap))
        else:
            expanded_units.append(unit)

    chunks: list[Chunk] = []
    current_units: list[str] = []
    chunk_index = 0

    def seal_chunk():
        nonlocal chunk_index
        text = " ".join(current_units).strip()
        if text:
            chunks.append(Chunk(text=text, source=source, page=page, chunk_index=chunk_index))
            chunk_index += 1

    for unit in expanded_units:
        if not current_units:
            current_units.append(unit)
            continue

        candidate_text = " ".join(current_units) + " " + unit
        if len(candidate_text) <= chunk_size:
            current_units.append(unit)
            continue

        seal_chunk()

        overlap_units = []
        overlap_len = 0
        for u in reversed(current_units):
            if overlap_len + len(u) > chunk_overlap:
                break
            overlap_units.insert(0, u)
            overlap_len += len(u) + 1

        current_units = overlap_units + [unit]

    seal_chunk()

    return chunks


def chunk_document_recursive(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Compositional wrapper: split into paragraph/sentence units, then
    pack them using fixed-size, character-count-based packing.
    """
    if not document.text.strip():
        return []
    units = split_text_paragraph_sentence(document.text, chunk_size)
    return pack_units_fixed_size(units, document.source, document.page, chunk_size, chunk_overlap)