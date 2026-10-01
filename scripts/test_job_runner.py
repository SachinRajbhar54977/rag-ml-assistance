# scripts/test_job_runner.py
import json
import time

from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.job_runner import start_job, get_job


def main():
    with open("data/eval/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    config = {
        "loader": "pymupdf",
        "splitter": "paragraph_sentence",
        "chunker": "fixed_size_char",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "distance_threshold": 1.0,
        "k": 5,
    }

    pdf_path = str(RAW_DATA_DIR / "machine_learning.pdf")

    print("Starting job...")
    job_id = start_job(pdf_path, eval_set, config)
    print(f"Job started: {job_id}")

    # Poll until the job finishes
    while True:
        job = get_job(job_id)
        print(f"  status: {job['status']}")

        if job["status"] in ("done", "failed"):
            break

        time.sleep(1)

    if job["status"] == "done":
        result = job["result"]
        print("\n=== RESULT ===")
        print(f"Accuracy: {result['passed']}/{result['total']} ({result['accuracy']*100:.1f}%)")
        print("\nBy category:")
        for cat, stats in result["categories"].items():
            rate = stats["passed"] / stats["total"] * 100
            print(f"  - {cat:<25} {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    else:
        print("\n=== JOB FAILED ===")
        print(job["error"])


if __name__ == "__main__":
    main()