import streamlit as st
import os
from rag_app import RAG

# Directory to save uploaded PDFs
UPLOAD_DIR = "uploaded_pdfs"
VECTOR_DBB_COLLECTION_NAME = "Earnings_Call_Transcripts_2"


os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("PDF Upload App")
uploaded_files = st.file_uploader("Drop PDF files here", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name} to {UPLOAD_DIR}")



# Import RAG system from RAG.py
# Initialize RAG system with uploaded PDFs
rag = RAG(vector_collection_name=VECTOR_DBB_COLLECTION_NAME)

existing_pdfs = set(os.listdir(UPLOAD_DIR))
new_pdfs = set(f.name for f in uploaded_files) if uploaded_files else set()
pdfs_to_process = list(new_pdfs - existing_pdfs)
print('here')
print(existing_pdfs)
print(pdfs_to_process)

if pdfs_to_process:
    # Add only new PDFs to the vector database
    rag.read_pdfs(pdfs_to_process)
    rag.add_to_vector_db()
else:
    rag.read_pdfs()
    rag.add_to_vector_db()



# UI layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.header("PDF Uploads")
    if uploaded_files:
        st.write("Uploaded PDFs:")
        for f in uploaded_files:
            st.write(f.name)
    else:
        st.write("No PDFs uploaded yet.")

with col2:
    st.header("RAG Chat Interface")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your PDFs:")
    if st.button("Send") and user_input:
        answer = rag.answer_query(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("RAG", answer))

    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {msg}")