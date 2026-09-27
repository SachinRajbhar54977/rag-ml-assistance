
from collections import Counter

from rag_assistant.ingestion.pdf_loader import Document


def detect_repeating_headers(
    documents: list[Document],
    min_frequency: float = 0.3,
) -> set[str]:
    """
    Scan the first line of every document's text.

    Return the set of lines whose frequency across all pages
    is greater than or equal to min_frequency.

    Args:
        documents: List of page-level Document objects.
        min_frequency: Minimum fraction of pages on which
            a line must appear.

    Returns:
        A set of repeating header candidates.
    """

    if not documents:
        return set()

    first_lines = []

    for doc in documents:
        if not doc.text or not doc.text.strip():
            continue

        first_line = doc.text.split("\n", 1)[0].strip()

        if first_line:
            first_lines.append(first_line)

    counts = Counter(first_lines)

    minimum_count = min_frequency * len(documents)

    headers = {
        line
        for line, count in counts.items()
        if count >= minimum_count
    }

    return headers
def strip_known_headers(text: str, headers: set[str]) -> str:
    """
    Remove lines from `text` that match a known header/footer, or that
    are bare page numbers — only checked within the first 2 and last 2
    lines of the page.
    """
    lines = text.splitlines()
    cleaned_lines = []

    total_lines = len(lines)

    for index, line in enumerate(lines):
        stripped_line = line.strip()

        is_edge_line = (
            index < 2 or
            index >= total_lines - 2
        )

        if is_edge_line and (
            stripped_line in headers or stripped_line.isdigit()
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_documents(documents: list[Document], headers: set[str]) -> list[Document]:
    """
    Apply header/page-number stripping to every document, returning
    new Document objects with cleaned text (originals are left untouched).
    """
    cleaned = []
    for doc in documents:
        cleaned_text = strip_known_headers(doc.text, headers)
        cleaned.append(Document(text=cleaned_text, source=doc.source, page=doc.page))
    return cleaned