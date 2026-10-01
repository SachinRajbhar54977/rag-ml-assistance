from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.prompting.prompt_builder import generate_answer_or_shortcircuit, build_prompt, format_answer_with_citations
from rag_assistant.generation.llm import generate_answer


def evaluate(retriever: Retriever, eval_set: list[dict], k: int = 5, llm_model_name: str | None = None) -> dict:
    """
    Run every eval question through the retriever and return a summary
    dict with overall accuracy, per-category breakdown, and failures.
    """
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

        generated_answer = None
        if llm_model_name is not None:
            shortcircuit = generate_answer_or_shortcircuit(question, retrieval_results)
            if shortcircuit is not None:
                generated_answer = shortcircuit
            else:
                prompt = build_prompt(question, retrieval_results)
                raw_answer = generate_answer(prompt, model_name=llm_model_name)
                generated_answer = format_answer_with_citations(raw_answer, retrieval_results)

        results.append({
            "question": question,
            "category": category,
            "should_find_answer": should_find,
            "expected_page": expected_page,
            "retrieved_pages": retrieved_pages,
            "passed": passed,
            "generated_answer": generated_answer,
        })

    

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])

    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "passed": 0})
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    return {
        "accuracy": total_passed / total if total > 0 else 0.0,
        "total": total,
        "passed": total_passed,
        "categories": categories,
        "details": results,
    }

