import streamlit as st
from rag_chain import generate_answer

st.set_page_config(
    page_title="RAG Knowledge Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Knowledge Bot")
st.write("Ask questions based on the company employee handbook.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching the knowledge base..."):
            answer = generate_answer(question)

        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")