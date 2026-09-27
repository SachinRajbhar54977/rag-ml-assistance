# scripts/compare_extraction.py

from pathlib import Path

import fitz
import pdfplumber
from pypdf import PdfReader

from rag_assistant.config import RAW_DATA_DIR


PAGE_NUMBER = 101  # 1-indexed, matching your Document.page convention


def extract_with_pypdf(path: Path, page_number: int) -> str:
    """Extract one page using pypdf."""
    reader = PdfReader(path)

    # pypdf uses 0-based indexing
    page = reader.pages[page_number - 1]

    return page.extract_text() or ""


def extract_with_pdfplumber(path: Path, page_number: int) -> str:
    """Extract one page using pdfplumber."""
    with pdfplumber.open(path) as pdf:
        # pdfplumber pages are also 0-indexed
        page = pdf.pages[page_number - 1]

        return page.extract_text() or ""


def extract_with_pymupdf(path: Path, page_number: int) -> str:
    """Extract one page using PyMuPDF."""
    doc = fitz.open(path)

    try:
        # PyMuPDF pages are 0-indexed
        page = doc[page_number - 1]

        return page.get_text() or ""

    finally:
        doc.close()


if __name__ == "__main__":
    path = RAW_DATA_DIR / "machine_learning.pdf"

    print("=== pypdf ===")
    print(extract_with_pypdf(path, PAGE_NUMBER)[:500])

    print("\n=== pdfplumber ===")
    print(extract_with_pdfplumber(path, PAGE_NUMBER)[:500])

    print("\n=== pymupdf (fitz) ===")
    print(extract_with_pymupdf(path, PAGE_NUMBER)[:500])