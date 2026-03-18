import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import textwrap

# -------------------------------
# Page Config (Fancy UI)
# -------------------------------
st.set_page_config(
    page_title="Offline PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# UI Header
# -------------------------------
st.markdown(
    "<h2 style='text-align:center;'>🤖 Offline Conversational PDF Chatbot</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Ask questions from your PDF (No Internet Required)</p>",
    unsafe_allow_html=True
)

# -------------------------------
# Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# -------------------------------
# PDF Upload
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# -------------------------------
# Functions
# -------------------------------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def chunk_text(text, size=400):
    return textwrap.wrap(text, size)

# -------------------------------
# Process PDF
# -------------------------------
if uploaded_file:
    raw_text = extract_text(uploaded_file)
    chunks = chunk_text(raw_text)

    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("✅ PDF processed successfully!")

# -------------------------------
# Chat Input
# -------------------------------
query = st.chat_input("Ask something...")

# -------------------------------
# Answer Generation
# -------------------------------
if query and st.session_state.index:
    query_embedding = model.encode([query])

    D, I = st.session_state.index.search(
        np.array(query_embedding), k=3
    )

    retrieved_text = " ".join(
        [st.session_state.chunks[i] for i in I[0]]
    )

    answer = (
        "Based on the document, here is the relevant information:\n\n"
        + retrieved_text
    )

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

# -------------------------------
# Display Chat (Fancy)
# -------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
