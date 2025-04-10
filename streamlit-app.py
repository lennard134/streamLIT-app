import re
import openai
import hashlib
import time
import os
import random
import requests
import torch
import base64
_ = torch.__file__ 
import pickle 
import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from scipy.spatial.distance import cosine

import pymupdf

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from streamlit_pdf_reader import pdf_reader
from huggingface_hub import InferenceClient


# ---- Config ----
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])  # Split screen

# ---- Set PDF File (Preloaded) ----

# def pdf_to_text(pdf_path):
#     """Extracts text from a PDF and returns it as a single long string."""
#     doc = pymupdf.open(pdf_path)
#     text = " ".join(page.get_text("text") for page in doc)
#     return re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces/newlines

#chunk split function
# def split_into_chunks(text, chunk_size, overlap=0.05):
#     """Splits text into chunks of given size with specified overlap."""
#     overlap_size = int(chunk_size * overlap)
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap_size  # Move forward with overlap

#         if end >= len(text):  # Stop if reaching the end
#             break

    # return chunks


# # document parse function
# def document_parsing(file_path, chunk_size):
#     """
#     Document parser, processes uploaded document and splits text into chunks for a given chunksize.
#     """
#     loader = PyMuPDFLoader(file_path)
#     pages = loader.load()
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_size/10, length_function=len)
#     return text_splitter.split_documents(pages)
    
# Caching model load and embedding
# @st.cache_resource
# def load_model():
#     return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)

@st.cache_data
def get_chunks_and_embeddings(pdf_path, chunk_size=512):
    # data_string = pdf_to_text(pdf_path)
    # chunks = split_into_chunks(data_string, chunk_size)
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    # model = load_model()
    embeddings = np.load('embeddings.npy')
    print(len(embeddings[0]))
    return chunks, embeddings

@st.cache_data(show_spinner="Processing PDF...")
def download_pdf_from_url() -> bytes:
    response = "https://drive.google.com/file/d/1MarcPdd5bLAVpyoTkU20m_fDVRTy0yyq/preview?usp=sharing"
    return response
    
# Session ID
if 'session_id' not in st.session_state:
    session_data = f"{time.time()}_{random.randint(0,int(1e6))}".encode()
    st.session_state['session_id'] = hashlib.sha256(session_data).hexdigest()[:16]

# Load & cache resources
pdf_path = "airplaneNoImage.pdf"
# model = load_model()
chunks, embeddings = get_chunks_and_embeddings(pdf_path)
st.session_state['file_path'] = pdf_path
st.session_state['chunks'] = chunks  # Small enough
#st.session_state['rawFile'] = "https://dl.dropbox.com/scl/fi/7esc4cp02p2kzuela3kgo/airplane.pdf?rlkey=dzmijzy8orn9bie73rmituaua&st=iws9qm3s&"


def get_embedding_with_retry(user_message, HF_client, max_retries=10, wait_time=1):
    retries = 0
    while retries < max_retries:
        try:
            question_embed = HF_client.feature_extraction(
                user_message,
                model="intfloat/multilingual-e5-large-instruct"
            )
            if question_embed is not None:
                return question_embed
            else:
                print(f"Request failed with status {response.status_code}: {response.text}")
                retries += 1
                wait_time = wait_time * 2  # Exponentially increase wait time
                print(f"Retrying... {retries}/{max_retries}")
                time.sleep(wait_time)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed due to error: {e}")
            retries += 1
            wait_time = wait_time * 2  # Exponentially increase wait time
            print(f"Retrying... {retries}/{max_retries}")
            time.sleep(wait_time)
    
    print("Max retries reached. Unable to get a valid response.")
    return None
# UI
with col1:
    st.header("💬 Chat with the PDF")

    # Secrets
    token = st.secrets["TOGETHER_API_TOKEN"]
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
    
    client = openai.OpenAI(api_key=token, base_url="https://api.together.xyz/v1")
    supabase_client: Client = create_client(url, key)
    HF_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    messages_box = st.container(height=600)
    for message in st.session_state["messages"]:
        messages_box.chat_message(message["role"]).write(message["content"])
        
    # User Input
    user_message = st.chat_input("Ask your question here")
    if user_message:
        # Embed user question
        # question_embedded = model.encode(user_message)
        question_embed = get_embedding_with_retry(user_message, HF_client)
        
        # similarities = model.similarity(embeddings, question_embedded)
        similarities = []
        for chunk_embedding in embeddings:
            similarity = 1 - cosine(question_embed, chunk_embedding)
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[::-1][:10]  # Indices of the top 10 similar chunks
        
        # Retrieve the top 10 most similar chunks based on the indices
        top_10_similar_chunks = [chunks[idx] for idx in top_indices]
        # tensor_values = similarities.view(-1)
        # top_k = torch.topk(tensor_values, k=10)
        # retrieved_context = ''.join([chunks[i] for i in top_k.indices])
        retrieved_context = ''.join(chunky for chunky in top_10_similar_chunks)
        st.session_state.messages.append({"role": "user", "content": user_message})
        if "messages" in st.session_state:  
            last_message = st.session_state.messages[-1]
            print(f'Last message: {last_message}')
        else:
            last_message = ''
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
        st.session_state.messages.append({"role": "RetrievedChunks", "content": retrieved_context})
        st.session_state.messages.append({"role": "assistant", "content": response_text})

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
#     # Get and cache PDF content
    pdf_bytes = download_pdf_from_url()
    
#     # If your pdf_reader can handle bytes:
    pdf_reader(pdf_bytes)
#     # pdf_reader(st.session_state['rawFile'])


