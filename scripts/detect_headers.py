# scripts/detect_headers.py
from collections import Counter
from rag_assistant.ingestion.pdf_loader import load_pdf, Document
from rag_assistant.config import RAW_DATA_DIR


def detect_repeating_headers(documents: list[Document], min_frequency: float = 0.03) -> set[str]:
    """
    Scan the first line of every document's text. Return the set of
    lines whose frequency across all pages exceeds `min_frequency`,
    treated as running headers/footers.
    """
    first_lines = []

    for doc in documents:
        if not doc.text.strip():
            continue  # skip empty pages, nothing to extract a first line from

        first_line = doc.text.split("\n", 1)[0].strip()

        if not first_line:
            continue  # guard against a page whose first line is blank

        first_lines.append(first_line)

    line_counts = Counter(first_lines)
    threshold = min_frequency * len(documents)

    headers = {line for line, count in line_counts.items() if count >= threshold}
    return headers


def main():
    docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
    headers = detect_repeating_headers(docs)

    print(f"Detected {len(headers)} repeating header candidates:")
    for h in headers:
        print(f"  {h!r}")


if __name__ == "__main__":
    main()