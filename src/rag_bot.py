from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 2. LOAD CHROMADB
# ==========================================

print("Loading vector database...")

vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


# ==========================================
# 3. LOAD LANGUAGE MODEL
# ==========================================

print("Loading language model...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)


# ==========================================
# 4. START CHATBOT
# ==========================================

print("\n======================================")
print("        RAG KNOWLEDGE BOT")
print("======================================")
print("Type 'exit' to stop the chatbot.\n")


while True:

    # --------------------------------------
    # Get user question
    # --------------------------------------

    query = input("Ask a question: ")

    if query.lower().strip() == "exit":
        print("\nGoodbye! 👋")
        break

    if not query.strip():
        print("Please enter a question.\n")
        continue


    # ======================================
    # 5. SEARCH CHROMADB WITH SCORE
    # ======================================

    results = vector_store.similarity_search_with_score(
        query,
        k=1
    )


    # ======================================
    # 6. GET BEST RESULT
    # ======================================

    document, distance = results[0]


    # ======================================
    # 7. DISPLAY RETRIEVED INFORMATION
    # ======================================

    print("\n========== RETRIEVED CONTEXT ==========")

    print(document.page_content)

    print(
        "Source:",
        document.metadata.get("source")
    )

    print(
        "Similarity distance:",
        distance
    )

    print("========================================")


    # ======================================
    # 8. CHECK RELEVANCE
    # ======================================

    # ChromaDB returns a distance.
    # Smaller distance means more similar.
    #
    # If the distance is too large,
    # the question is probably not
    # covered by our knowledge base.

    MAX_DISTANCE = 0.90


    if distance > MAX_DISTANCE:

        print("\nBot:")
        print(
            "I don't know based on the provided knowledge base."
        )

        print("\n--------------------------------------\n")

        continue


    # ======================================
    # 9. CREATE CONTEXT
    # ======================================

    context = document.page_content


    # ======================================
    # 10. CREATE RAG PROMPT
    # ======================================

    prompt = f"""Use ONLY the information in the Context to answer the Question.

If the answer is not present in the Context, say:
I don't know based on the provided knowledge base.

Context:
{context}

Question:
{query}

Answer:
"""


    # ======================================
    # 11. GENERATE ANSWER
    # ======================================

    response = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False
    )


    # ======================================
    # 12. GET ANSWER
    # ======================================

    answer = response[0]["generated_text"].strip()


    # ======================================
    # 13. DISPLAY ANSWER
    # ======================================

    print("\nBot:")
    print(answer)

    print("\n--------------------------------------\n")