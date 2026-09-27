# scripts/run_evaluation.py
import json
from pathlib import Path

from rag_assistant.config import PROCESSED_DATA_DIR
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(retriever: Retriever, eval_set: list[dict], k: int = 5) -> list[dict]:
    """
    Run every eval question through the retriever and record results.
    """
    results = []

    for item in eval_set:
        question = item["question"]
        expected_page = item.get("expected_page")
        category = item.get("category", "uncategorized")
        should_find = item.get("should_find_answer", True)

        # Retrieve top-k results — correct keyword name is `k`, not `top_k`
        retrieval_results = retriever.retrieve(question, k=k)

        # 1. Answer presence check (refusal / threshold check)
        found_any = len(retrieval_results) > 0
        retrieval_success = (found_any == should_find)

        # Each item is a RetrievalResult(chunk=Chunk(...), distance=...)
        # so the real path to a page number is r.chunk.page — no dict fallback needed.
        retrieved_pages = [r.chunk.page for r in retrieval_results]

        # 2. Page hit check (only applicable when an answer should be found)
        page_hit = False
        if should_find and expected_page is not None:
            page_hit = any(
                p is not None and abs(p - expected_page) <= 1
                for p in retrieved_pages
            )

        # Overall pass criteria:
        # - out-of-scope (should_find=False): pass if nothing retrieved
        # - in-scope (should_find=True): pass if retrieved AND correct page found
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
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation set not found at {eval_path}")

    eval_set = load_eval_set(str(eval_path))

    index_path = PROCESSED_DATA_DIR / "faiss_index.bin"
    metadata_path = PROCESSED_DATA_DIR / "metadata.pkl"

    store = VectorStore(dimension=384)
    store.load(str(index_path), str(metadata_path))
    retriever = Retriever(store, distance_threshold=1.0)

    results = evaluate(retriever, eval_set, k=5)

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    overall_accuracy = (total_passed / total * 100) if total > 0 else 0.0

    print("=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {total_passed}/{total} ({overall_accuracy:.1f}%)\n")

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print("Breakdown by Category:")
    print("-" * 50)
    for cat, stats in categories.items():
        pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        print(f"  - {cat:<25} {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n" + "=" * 70)
        print("FAILED CASES")
        print("=" * 70)
        for idx, f in enumerate(failures, 1):
            print(f"{idx}. Question: {f['question']}")
            print(f"   Category: {f['category']}")
            print(f"   Expected Page: {f['expected_page']} | Retrieved Pages: {f['retrieved_pages']}")
            print(f"   Should Find Answer: {f['should_find_answer']} | Found Any: {f['found_any']}")
            print("-" * 50)


if __name__ == "__main__":
    main()