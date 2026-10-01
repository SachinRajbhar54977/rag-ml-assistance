from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf

docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")

doc_25 = next(d for d in docs if d.page == 25)
print(doc_25.text[:300])