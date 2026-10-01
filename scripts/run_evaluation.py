# scripts/run_evaluation.py
import json
from pathlib import Path

from rag_assistant.config import PROCESSED_DATA_DIR
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.evaluation.evaluator import evaluate


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # Set to a real model name to also generate answers (slower),
    # or None for fast, retrieval-only evaluation.
    #llm_model_name = None  # try "Qwen/Qwen2.5-1.5B-Instruct" for the generation test
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    result = evaluate(retriever, eval_set, k=5, llm_model_name=llm_model_name)

    total = result["total"]
    total_passed = result["passed"]
    overall_accuracy = result["accuracy"] * 100

    print("=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {total_passed}/{total} ({overall_accuracy:.1f}%)\n")

    print("Breakdown by Category:")
    print("-" * 50)
    for cat, stats in result["categories"].items():
        pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        print(f"  - {cat:<25} {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")

    if llm_model_name is not None:
        print("\nSample generated answer (question 1):")
        print(result["details"][0]["generated_answer"])

    failures = [r for r in result["details"] if not r["passed"]]
    if failures:
        print("\n" + "=" * 70)
        print("FAILED CASES")
        print("=" * 70)
        for idx, f in enumerate(failures, 1):
            print(f"{idx}. Question: {f['question']}")
            print(f"   Category: {f['category']}")
            print(f"   Expected Page: {f['expected_page']} | Retrieved Pages: {f['retrieved_pages']}")
            print("-" * 50)


if __name__ == "__main__":
    main()