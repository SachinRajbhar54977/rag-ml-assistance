import time
from rag_assistant.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from rag_assistant.vectorstore.faiss_store import VectorStore
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.evaluation.evaluator import evaluate

print("Loading saved index...")
t0 = time.time()
store = VectorStore(dimension=384)
store.load(str(PROCESSED_DATA_DIR / "faiss_index.bin"), str(PROCESSED_DATA_DIR / "metadata.pkl"))
print(f"  done in {time.time() - t0:.1f}s")

retriever = Retriever(store, distance_threshold=1.0)

small_eval_set = [
    {
        "question": "What is a partial dependence plot?",
        "expected_page": 117,
        "category": "definition",
        "should_find_answer": True,
    },
    {
        "question": "How do you calculate the exact p-value for a two-sample t-test?",
        "expected_page": None,
        "category": "out_of_scope",
        "should_find_answer": False,
    },
]

print("\nStarting evaluate() with llm_model_name set...")
t0 = time.time()
result = evaluate(
    retriever,
    small_eval_set,
    k=5,
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
)
print(f"  evaluate() finished in {time.time() - t0:.1f}s")

print(f"\nAccuracy: {result['passed']}/{result['total']}")
for item in result["details"]:
    print(f"\n--- {item['question']} ---")
    print(item["generated_answer"])