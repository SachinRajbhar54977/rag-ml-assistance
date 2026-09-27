import numpy as np
from rag_assistant.embedding.embedder import embed_documents

texts = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
]

vectors = embed_documents(texts)

for i, v in enumerate(vectors):
    v_array = np.array(v)
    norm = np.linalg.norm(v_array)
    print(f"Vector {i} raw norm: {norm}")