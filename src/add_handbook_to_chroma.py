from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load employee handbook

loader = TextLoader(
    "data/documents/employee_handbook.txt"
)

documents = loader.load()

print("Documents loaded:", len(documents))


# 2. Split the document into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Load embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Open existing ChromaDB

vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


# 5. Add handbook chunks to ChromaDB

vector_store.add_documents(chunks)

print("Employee handbook added to ChromaDB successfully!")
print("Documents added:", len(chunks))