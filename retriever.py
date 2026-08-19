import pickle
import faiss


# --------------------------------------------------
# 1. Load FAISS index
# --------------------------------------------------

index = faiss.read_index(
    "vectorstore/company_docs.index"
)


# --------------------------------------------------
# 2. Load TF-IDF vectorizer
# --------------------------------------------------

with open(
    "vectorstore/vectorizer.pkl",
    "rb"
) as file:

    vectorizer = pickle.load(file)


# --------------------------------------------------
# 3. Load document chunks
# --------------------------------------------------

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as file:

    chunks = pickle.load(file)


# --------------------------------------------------
# 4. Function to retrieve relevant documents
# --------------------------------------------------

def retrieve_documents(query, k=2):

    # Convert user's question into TF-IDF vector
    query_vector = vectorizer.transform(
        [query]
    )

    # Convert to FAISS-compatible format
    query_vector = query_vector.toarray().astype(
        "float32"
    )

    # Search FAISS
    distances, indices = index.search(
        query_vector,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx != -1:

            results.append({
                "content": chunks[idx].page_content,
                "metadata": chunks[idx].metadata,
                "distance": float(distance)
            })

    return results


