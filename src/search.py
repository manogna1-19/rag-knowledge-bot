from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load ChromaDB
print("Loading ChromaDB...")

vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


# Ask user for a question
query = input("\nEnter your question: ")


# Search ChromaDB
results = vector_store.similarity_search(
    query,
    k=3
)


# Display results
print("\n========== SEARCH RESULTS ==========")

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)

    print("Metadata:", document.metadata)

print("\n====================================")