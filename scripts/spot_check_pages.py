# scripts/spot_check_pages.py
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.config import RAW_DATA_DIR


def count_long_tokens(text: str, threshold: int = 25) -> int:
    tokens = text.split()
    long_count = 0
    for token in tokens:
        cleaned = token.strip(".,;:!?()[]\"'")
        if len(cleaned) > threshold:
            long_count += 1
    return long_count


def main():
    docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")

    affected = [(d.page, count_long_tokens(d.text), d) for d in docs if count_long_tokens(d.text) > 0]
    affected.sort(key=lambda x: x[1])

    samples_to_check = [affected[0], affected[len(affected) // 2], affected[-2]]

    for page_num, count, doc in samples_to_check:
        print(f"\n=== Page {page_num} (long tokens: {count}) ===")
        print(doc.text[:400])


if __name__ == "__main__":
    main()