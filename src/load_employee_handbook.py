from langchain_community.document_loaders import TextLoader

file_path = "data/documents/employee_handbook.txt"

loader = TextLoader(file_path)

documents = loader.load()

print("Number of documents:", len(documents))

for document in documents:
    print("\n--- Document Content ---")
    print(document.page_content)

    print("\n--- Metadata ---")
    print(document.metadata)