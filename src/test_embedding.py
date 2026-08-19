from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Java is a programming language.",
    "Python is commonly used for machine learning.",
    "RAG combines document retrieval with a language model."
]

embeddings = model.encode(texts)

print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])