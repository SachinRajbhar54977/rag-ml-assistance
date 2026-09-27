# scripts/test_header_cleaning.py
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, strip_known_headers
from rag_assistant.config import RAW_DATA_DIR


def main():
    docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
    headers = detect_repeating_headers(docs, min_frequency=0.05)

    # --- Check 1: page 101, the recurring test case ---
    page_101 = next(d for d in docs if d.page == 101)

    print("=== PAGE 101 BEFORE ===")
    print(page_101.text[:200])

    cleaned_101 = strip_known_headers(page_101.text, headers)

    print("\n=== PAGE 101 AFTER ===")
    print(cleaned_101[:200])

    # --- Check 2: locate the table page with a looser search ---
    print("\n=== SEARCHING FOR TABLE PAGE ===")
    for d in docs:
        if "season" in d.text.lower():
            print(f"\nDocument.page = {d.page}")
            print(d.text[:400])
            print("---")

    table_doc = next(d for d in docs if d.page == 50)

    print("=== PAGE 50 (TABLE) BEFORE ===")
    print(table_doc.text[:500])

    cleaned_table = strip_known_headers(table_doc.text, headers)

    print("\n=== PAGE 50 (TABLE) AFTER ===")
    print(cleaned_table[:500])

if __name__ == "__main__":
    main()