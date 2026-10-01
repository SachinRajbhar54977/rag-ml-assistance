from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.pipeline_builder import build_pipeline
from rag_assistant.retrieval.retriever import Retriever

store = build_pipeline(
    pdf_path=str(RAW_DATA_DIR / "machine_learning.pdf"),
    loader_name="pymupdf",
    splitter_name="paragraph_sentence",
    chunker_name="fixed_size_char",
    embedding_model_name="all-MiniLM-L6-v2",
)

print(f"Built store with {store.index.ntotal} vectors")

retriever = Retriever(store, distance_threshold=1.0)
results = retriever.retrieve("What is a partial dependence plot?", k=5)
for r in results:
    print(f"page={r.chunk.page}, distance={r.distance:.4f}")