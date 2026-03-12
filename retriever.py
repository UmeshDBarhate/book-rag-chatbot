from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Load embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Cache vector store
# -------------------------------


@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "document_file_db",
        embeddings,
        allow_dangerous_deserialization=True
    )


vector_store = load_vector_store()

# -------------------------------
# Load LLM
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# -------------------------------
# UI
# -------------------------------
st.title("📚 Book Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask something about the book")

if prompt:

    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # -------------------------------
    # Better retrieval (MMR)
    # -------------------------------
    docs = vector_store.max_marginal_relevance_search(
        prompt,
        k=5,
        fetch_k=20
    )

    context = "\n\n".join([d.page_content for d in docs])

    # -------------------------------
    # Source extraction
    # -------------------------------
    sources = []
    for d in docs:
        page = d.metadata.get("page", "unknown")
        sources.append(f"Page {page}")

    sources_text = ", ".join(set(sources))

    # -------------------------------
    # Better RAG prompt
    # -------------------------------
    final_prompt = f"""
You are a helpful assistant answering questions from a book.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I don't know based on the book."

Context:
{context}

Question:
{prompt}

Answer:
"""

    # -------------------------------
    # Streaming response
    # -------------------------------
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(final_prompt):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        st.caption(f"Sources: {sources_text}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
