from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "The stock market crashed today.",
]

embeddings = model.encode(sentences)

# Shape of embeddings
print("Embedding shape:", embeddings.shape)

# Cosine similarity
similarity_01 = util.cos_sim(embeddings[0], embeddings[1])
similarity_02 = util.cos_sim(embeddings[0], embeddings[2])

print("Similarity (sentence 0 vs 1):", similarity_01.item())
print("Similarity (sentence 0 vs 2):", similarity_02.item())  