import time
from rag_assistant.embedding.embedder import embed_query, _loaded_models

# First call for MiniLM — should load from disk/HF
t0 = time.time()
embed_query("test query", model_name="all-MiniLM-L6-v2")
print(f"First MiniLM call: {time.time() - t0:.3f}s, cache size: {len(_loaded_models)}")

# Second call for MiniLM — should be fast, using the cache
t0 = time.time()
embed_query("another query", model_name="all-MiniLM-L6-v2")
print(f"Second MiniLM call: {time.time() - t0:.3f}s, cache size: {len(_loaded_models)}")

# First call for a DIFFERENT model — should load fresh (slower), cache grows
t0 = time.time()
embed_query("test query", model_name="all-mpnet-base-v2")
print(f"First mpnet call: {time.time() - t0:.3f}s, cache size: {len(_loaded_models)}")

# Second call for mpnet — should be fast again
t0 = time.time()
embed_query("another query", model_name="all-mpnet-base-v2")
print(f"Second mpnet call: {time.time() - t0:.3f}s, cache size: {len(_loaded_models)}")