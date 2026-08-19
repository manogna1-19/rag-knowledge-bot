from pypdf import PdfReader

pdf_path = "data/documents/sample.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

for i, page in enumerate(reader.pages):
    text = page.extract_text()

    print(f"\n--- Page {i + 1} ---")
    print(text)