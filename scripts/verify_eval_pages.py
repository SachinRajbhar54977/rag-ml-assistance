from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.ingestion.pdf_loader import load_pdf
from rag_assistant.cleaning.header_cleaner import detect_repeating_headers, clean_documents

docs = load_pdf(RAW_DATA_DIR / "machine_learning.pdf")
headers = detect_repeating_headers(docs, min_frequency=0.05)
cleaned_docs = clean_documents(docs, headers)

keywords_by_question = {
    "PDP definition": "partial dependence plot",
    "ALE vs PDP": "accumulated local effects",
    "SBRL": "scalable bayesian rule list",
    "RuleFit": "rulefit",
    "Cook's distance": "cook's distance",
    "correlated features": "correlated",  # too generic on its own — think about this one
    "RuleFit rephrase": "rulefit",
}

for label, keyword in keywords_by_question.items():
    print(f"\n=== {label} (searching: {keyword!r}) ===")
    for d in cleaned_docs:
        if keyword.lower() in d.text.lower():
            print(f"  page {d.page}: {d.text[:100]!r}")