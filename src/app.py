import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG Knowledge Bot",
    page_icon="📚",
    layout="centered"
)


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# LOAD CHROMADB
# ==========================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()

    return Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )


# ==========================================
# INITIALIZE
# ==========================================

vector_store = load_vector_store()


# ==========================================
# TITLE
# ==========================================

st.title("📚 RAG Knowledge Bot")

st.write(
    "Upload documents and ask questions about your knowledge base."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )


    # ======================================
    # PROCESS UPLOADED FILE
    # ======================================

    if uploaded_file is not None:

        if st.button("Add to Knowledge Base"):

            with st.spinner("Processing document..."):

                documents = []


                # ------------------------------
                # TXT FILE
                # ------------------------------

                if uploaded_file.name.lower().endswith(".txt"):

                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": uploaded_file.name
                            }
                        )
                    )


                # ------------------------------
                # PDF FILE
                # ------------------------------

                elif uploaded_file.name.lower().endswith(".pdf"):

                    pdf_reader = PdfReader(
                        uploaded_file
                    )

                    for page_number, page in enumerate(
                        pdf_reader.pages
                    ):

                        text = page.extract_text()

                        if text:

                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "page": page_number + 1
                                    }
                                )
                            )


                # ==================================
                # CHECK DOCUMENT
                # ==================================

                if not documents:

                    st.error(
                        "Could not extract text from this file."
                    )

                else:

                    # ==================================
                    # SPLIT TEXT INTO CHUNKS
                    # ==================================

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    chunks = splitter.split_documents(
                        documents
                    )


                    # ==================================
                    # ADD TO CHROMADB
                    # ==================================

                    vector_store.add_documents(
                        chunks
                    )


                    st.success(
                        f"Document added successfully! "
                        f"{len(chunks)} chunks stored."
                    )


    st.divider()


    # ==========================================
    # KNOWLEDGE BASE INFORMATION
    # ==========================================

    st.header("📚 Knowledge Base")

    st.write(
        "Documents stored in ChromaDB can be searched "
        "using semantic similarity."
    )

    st.divider()

    st.subheader("Existing Documents")

    st.write("📄 knowledge.txt")
    st.write("📄 employee_handbook.txt")

    st.divider()

    st.write("🔎 ChromaDB — semantic search")
    st.write("🧠 Sentence Transformers — embeddings")


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.chat_input(
    "Ask a question about the knowledge base..."
)


if question:

    # ======================================
    # DISPLAY QUESTION
    # ======================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # ======================================
    # SEARCH CHROMADB
    # ======================================

    with st.spinner("Searching knowledge base..."):

        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=3
        )


    # ======================================
    # RELEVANCE THRESHOLD
    # ======================================

    MIN_RELEVANCE_SCORE = 0.10


    relevant_results = [
        (document, score)
        for document, score in results
        if score >= MIN_RELEVANCE_SCORE
    ]


    # ======================================
    # NO RELEVANT INFORMATION
    # ======================================

    if not relevant_results:

        answer = (
            "I don't know based on the provided knowledge base."
        )

        sources = []


    # ======================================
    # RELEVANT INFORMATION FOUND
    # ======================================

    else:

        context_parts = []
        sources = []


        for document, score in relevant_results:

            context_parts.append(
                document.page_content
            )

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            if source not in sources:

                sources.append(source)


        # --------------------------------------
        # COMBINE RETRIEVED INFORMATION
        # --------------------------------------

        answer = "\n\n".join(
            context_parts
        )


    # ======================================
    # DISPLAY ANSWER
    # ======================================

    with st.chat_message("assistant"):

        st.write(answer)


        st.divider()

        st.caption("📌 Sources")


        if sources:

            for source in sources:

                st.caption(source)

        else:

            st.caption(
                "No relevant source found."
            )


    # ======================================
    # SAVE RESPONSE
    # ======================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )