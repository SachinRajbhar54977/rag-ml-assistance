# scripts/run_evaluation_recursive.py
import json
from pathlib import Path

from rag_assistant.config import PROCESSED_DATA_DIR
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(retriever: Retriever, eval_set: list[dict], k: int = 5) -> list[dict]:
    results = []
    for item in eval_set:
        question = item["question"]
        expected_page = item.get("expected_page")
        category = item.get("category", "uncategorized")
        should_find = item.get("should_find_answer", True)

        retrieval_results = retriever.retrieve(question, k=k)
        found_any = len(retrieval_results) > 0
        retrieved_pages = [r.chunk.page for r in retrieval_results]

        page_hit = False
        if should_find and expected_page is not None:
            page_hit = any(p is not None and abs(p - expected_page) <= 1 for p in retrieved_pages)

        passed = (not should_find and not found_any) or (should_find and page_hit)

        results.append({
            "question": question,
            "category": category,
            "should_find_answer": should_find,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "found_any": found_any,
            "page_hit": page_hit,
            "passed": passed,
        })
    return results


def main():
    eval_path = Path("data/eval/eval_set.json")
    eval_set = load_eval_set(str(eval_path))

    # Only difference from run_evaluation.py: point at the recursive index
    index_path = PROCESSED_DATA_DIR / "faiss_index_recursive.bin"
    metadata_path = PROCESSED_DATA_DIR / "metadata_recursive.pkl"

    store = VectorStore(dimension=384)
    store.load(str(index_path), str(metadata_path))
    retriever = Retriever(store, distance_threshold=0.90)

    results = evaluate(retriever, eval_set, k=5)

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    print("=" * 70)
    print("EVALUATION RESULTS (RECURSIVE CHUNKING)")
    print("=" * 70)
    print(f"Overall Accuracy: {total_passed}/{total} ({total_passed/total*100:.1f}%)\n")

    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "passed": 0})
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    for cat, stats in categories.items():
        rate = stats["passed"] / stats["total"] * 100
        print(f"  - {cat:<25} {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFAILED CASES")
        for f in failures:
            print(f"- {f['question']} | expected={f['expected_page']} retrieved={f['retrieved_pages']}")


if __name__ == "__main__":
    main()