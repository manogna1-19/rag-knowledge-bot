from langchain_community.document_loaders import TextLoader

# Load the knowledge base
loader = TextLoader("data/knowledge.txt", encoding="utf-8")

documents = loader.load()

print("Number of documents loaded:", len(documents))

print("\nDocument content:\n")
print(documents[0].page_content)

print("\nDocument metadata:")
print(documents[0].metadata)