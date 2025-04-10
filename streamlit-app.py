import re
import openai
import hashlib
import time
import os
import random
import requests
import torch
_ = torch.__file__ 

import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader

import pymupdf

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from streamlit_pdf_reader import pdf_reader

# ---- Config ----
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])  # Split screen

# ---- Set PDF File (Preloaded) ----

def pdf_to_text(pdf_path):
    """Extracts text from a PDF and returns it as a single long string."""
    doc = pymupdf.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces/newlines

#chunk split function
def split_into_chunks(text, chunk_size, overlap=0.05):
    """Splits text into chunks of given size with specified overlap."""
    overlap_size = int(chunk_size * overlap)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap_size  # Move forward with overlap

        if end >= len(text):  # Stop if reaching the end
            break

    return chunks


# document parse function
def document_parsing(file_path, chunk_size):
    """
    Document parser, processes uploaded document and splits text into chunks for a given chunksize.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_size/10, length_function=len)
    return text_splitter.split_documents(pages)
    
# Caching model load and embedding
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)

@st.cache_data
def get_chunks_and_embeddings(pdf_path, chunk_size=512):
    data_string = pdf_to_text(pdf_path)
    chunks = split_into_chunks(data_string, chunk_size)
    model = load_model()
    embeddings = np.load('embeddings.npy')
    return chunks, embeddings

@st.cache_data(show_spinner="Processing PDF...")
def download_pdf_from_url() -> bytes:
    response = requests.get("https://dl.dropbox.com/scl/fi/7esc4cp02p2kzuela3kgo/airplane.pdf?rlkey=dzmijzy8orn9bie73rmituaua&st=iws9qm3s&")
    response.raise_for_status()
    return response.content
    
# Session ID
if 'session_id' not in st.session_state:
    session_data = f"{time.time()}_{random.randint(0,int(1e6))}".encode()
    st.session_state['session_id'] = hashlib.sha256(session_data).hexdigest()[:16]

# Load & cache resources
pdf_path = "airplaneNoImage.pdf"
model = load_model()
chunks, embeddings = get_chunks_and_embeddings(pdf_path)
st.session_state['file_path'] = pdf_path
st.session_state['chunks'] = chunks  # Small enough
#st.session_state['rawFile'] = "https://dl.dropbox.com/scl/fi/7esc4cp02p2kzuela3kgo/airplane.pdf?rlkey=dzmijzy8orn9bie73rmituaua&st=iws9qm3s&"

# UI
with col1:
    st.header("💬 Chat with the PDF")

    # Secrets
    token = st.secrets["TOGETHER_API_TOKEN"]
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    
    client = openai.OpenAI(api_key=token, base_url="https://api.together.xyz/v1")
    supabase_client: Client = create_client(url, key)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    messages_box = st.container(height=600)
    for msg in st.session_state["messages"]:
        messages_box.chat_message(msg["role"]).write(msg["content"])

    # User Input
    user_message = st.chat_input("Ask your question here")
    if user_message:
        # Embed user question
        question_embedded = model.encode(user_message)
        similarities = model.similarity(embeddings, question_embedded)
        tensor_values = similarities.view(-1)
        top_k = torch.topk(tensor_values, k=10)
        retrieved_context = ''.join([chunks[i] for i in top_k.indices])

        st.session_state.messages.append({"role": "user", "content": user_message})

        custom_prompt = f"""
                        You are a helpful assistant that based on retrieved documents returns a response that fits with the question of the user.
                        Your role is to:
                        1. Answer questions by the user using the provided retrieved documents.
                        2. Never generate information beyond what is retrieved from the document.
                        3. Use information provided by the user
                        Inputs:
                        - Retrieved Context: {retrieved_context}
                        - User Question: {user_message}
                        - Assitant previous response: {last_message}
                        Provide a constructive response that is to the point and as concise as possible. Answer only based on the information retrieved from the document and given by the detective.                        
                    """ 

        result = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "assistant", "content": custom_prompt}],
        )

        response_text = result.choices[0].message.content
        st.session_state.messages.extend([
            {"role": "RetrievedChunks", "content": retrieved_context},
            {"role": "assistant", "content": response_text}
        ])

        # Save to Supabase
        supabase_client.table("testEnvironment").insert({
            "session_id": st.session_state.session_id,
            "Question": user_message,
            "Answer": response_text
        }).execute()

        st.rerun()

    if st.button("Next question"):
        st.session_state["messages"] = []
        st.rerun()

with col2:
    # Get and cache PDF content
    pdf_bytes = download_pdf_from_url()
    
    # If your pdf_reader can handle bytes:
    pdf_reader(pdf_bytes)
    # pdf_reader(st.session_state['rawFile'])


