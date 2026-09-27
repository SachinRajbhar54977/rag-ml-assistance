# scripts/diagnose_cleaning.py
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.config import RAW_DATA_DIR


def count_long_tokens(text: str, threshold: int = 25) -> int:
    """
    Count whitespace-separated tokens longer than `threshold` characters.
    Trailing punctuation is stripped before measuring length, so a real
    word followed by a period/comma isn't miscounted as "too long".
    """
    tokens = text.split()
    long_count = 0
    for token in tokens:
        cleaned = token.strip(".,;:!?()[]\"'")
        if len(cleaned) > threshold:
            long_count += 1
    return long_count


def main():
    pdf_path = RAW_DATA_DIR / "machine_learning.pdf"
    docs = load_pdf(pdf_path)

    page_counts = []  # list of (page_number, long_token_count)
    for doc in docs:
        count = count_long_tokens(doc.text)
        page_counts.append((doc.page, count))

    clean_pages = [p for p, c in page_counts if c == 0]
    affected_pages = [(p, c) for p, c in page_counts if c > 0]

    print(f"Total pages analyzed: {len(page_counts)}")
    print(f"Clean pages (0 long tokens): {len(clean_pages)}")
    print(f"Affected pages (>0 long tokens): {len(affected_pages)}")

    if affected_pages:
        avg_count = sum(c for _, c in affected_pages) / len(affected_pages)
        print(f"Average long-token count on affected pages: {avg_count:.2f}")

        worst_page, worst_count = max(affected_pages, key=lambda x: x[1])
        print(f"\nWorst offender: page {worst_page} with {worst_count} long tokens")

        worst_doc = next(d for d in docs if d.page == worst_page)
        print(f"\n--- Sample text from page {worst_page} ---")
        print(worst_doc.text[:500])
    else:
        print("No affected pages found — extraction looks clean.")


if __name__ == "__main__":
    main()

    