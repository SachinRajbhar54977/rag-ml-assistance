
# src/rag_assistant/ingestion/pdf_loader.py

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from rag_assistant.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Document:
    text: str
    source: str
    page: int


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF and return one Document per page that has extractable text.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents: list[Document] = []
    empty_pages: list[int] = []

    with pymupdf.open(path) as pdf:

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            try:
                text = page.get_text("text")

            except Exception as exc:
                logger.warning(
                    "Failed to extract page %s of %s: %s",
                    page_number,
                    path.name,
                    exc,
                )
                empty_pages.append(page_number)
                continue

            if not text or not text.strip():
                empty_pages.append(page_number)
                continue

            documents.append(
                Document(
                    text=text,
                    source=path.name,
                    page=page_number,
                )
            )

        total_pages = len(pdf)

    if empty_pages:
        logger.warning(
            "%d of %d pages in %s had no extractable text: pages %s",
            len(empty_pages),
            total_pages,
            path.name,
            empty_pages,
        )

    logger.info(
        "Loaded %d pages with text from %s",
        len(documents),
        path.name,
    )

    return documents

