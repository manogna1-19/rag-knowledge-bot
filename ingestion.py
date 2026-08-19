import os
import pickle
import faiss

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# 1. Load company documents
# --------------------------------------------------

documents_path = "data/documents/employee_handbook.txt"

loader = TextLoader(
    documents_path,
    encoding="utf-8"
)

documents = loader.load()

print(f"Documents loaded: {len(documents)}")


# --------------------------------------------------
# 2. Split documents into smaller chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# --------------------------------------------------
# 3. Extract text from chunks
# --------------------------------------------------

texts = [
    chunk.page_content
    for chunk in chunks
]


# --------------------------------------------------
# 4. Create TF-IDF vectors
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english"
)

vectors = vectorizer.fit_transform(texts)

print("TF-IDF vectors created.")
print("Vector shape:", vectors.shape)


# --------------------------------------------------
# 5. Convert vectors for FAISS
# --------------------------------------------------

vectors = vectors.toarray().astype("float32")


# --------------------------------------------------
# 6. Create FAISS index
# --------------------------------------------------

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

print("FAISS index created.")
print("Vectors stored in FAISS:", index.ntotal)


# --------------------------------------------------
# 7. Create vectorstore directory
# --------------------------------------------------

os.makedirs(
    "vectorstore",
    exist_ok=True
)


# --------------------------------------------------
# 8. Save FAISS index
# --------------------------------------------------

faiss.write_index(
    index,
    "vectorstore/company_docs.index"
)


# --------------------------------------------------
# 9. Save vectorizer
# --------------------------------------------------

with open(
    "vectorstore/vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# --------------------------------------------------
# 10. Save document chunks
# --------------------------------------------------

with open(
    "vectorstore/chunks.pkl",
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


print("\nKnowledge base created successfully!")
print("Saved inside: vectorstore/")