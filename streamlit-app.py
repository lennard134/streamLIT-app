import re
from openai import OpenAI
import hashlib
import time
import random
import requests
import pickle 
import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine

from supabase import create_client, Client
from streamlit_pdf_reader import pdf_reader


# ---- Config ----
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])  # Split screen

# ---- Set PDF File (Preloaded) ----
@st.cache_data
def get_chunks_and_embeddings():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load('embeddings.npy')
    return chunks, embeddings

def query(API_URL, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Session ID
if 'session_id' not in st.session_state:
    session_data = f"{time.time()}_{random.randint(0,int(1e6))}".encode()
    st.session_state['session_id'] = hashlib.sha256(session_data).hexdigest()[:16]

# Load & cache resources
chunks, embeddings = get_chunks_and_embeddings()
#test comment
def get_embedding_with_retry(user_message, API_URL, max_retries=5, wait_time=2):
    retries = 0
    with st.spinner("Processing"):
        while retries < max_retries:
            try:
                question_embed = query(API_URL=API_URL,payload={
                    "inputs": user_message,
                })
                if question_embed is not None:
                    return question_embed
                else:
                    retries += 1
                    wait_time = wait_time + 2  # Exponentially increase wait time
                    print(f"Retrying... {retries}/{max_retries}")
                    time.sleep(wait_time)
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed due to error: {e}")
                retries += 1
                wait_time = wait_time + 2  # Exponentially increase wait time
                print(f"Retrying... {retries}/{max_retries}")
                time.sleep(wait_time)
        out = 'Bad Request: Please try again'
        return out

def get_model_response(user_message, HF_client, model_name, max_retries=5, wait_time=2):
    retries = 0
    with st.spinner("Processing"):
        while retries < max_retries:
            try:
                completion = HF_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    max_tokens=500,
                )
                if completion is not None:
                    return completion.choices[0].message.content
                else:
                    retries += 1
                    wait_time = wait_time + 2  # Exponentially increase wait time
                    print(f"Retrying... {retries}/{max_retries}")
                    time.sleep(wait_time)
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed due to error: {e}")
                retries += 1
                wait_time = wait_time + 2  # Exponentially increase wait time
                print(f"Retrying... {retries}/{max_retries}")
                time.sleep(wait_time)
        
        out = 'Bad Request: Please try again'
        return out

#UI




# model_name = "meta-llama/llama-3.2-3b-instruct" 
# base_url = "https://router.huggingface.co/novita/v3/openai"   

model_name = "meta-llama/llama-3.1-8b-instruct"
base_url = "https://router.huggingface.co/novita/v3/openai",

# model_name = "meta-llama/Llama-3.3-70B-Instruct"
# base_url="https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1"

st.session_state["MODEL_CHOSEN"] = True
    
with col1:
    
    st.header("💬 Assistant")

    # Secrets
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_TOKEN"]
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    HF_TOKEN = st.secrets["HF_API_TOKEN"]    
 
    supabase_client: Client = create_client(url, key)

    HF_client_LLM = OpenAI(
        base_url=base_url,
        api_key=HF_TOKEN,
    )
    
    authorization = "Bearer " + HF_TOKEN
    API_URL = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-large-instruct/pipeline/feature-extraction"
    headers = {
        "Authorization": authorization,
    }
    
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
        question_embed = get_embedding_with_retry(user_message, API_URL)
        similarities = []
        for chunk_embedding in embeddings:
            similarity = 1 - cosine(question_embed, chunk_embedding)
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[::-1][:5]  # Indices of the top 10 similar chunks
        
        # Retrieve the top 10 most similar chunks based on the indices
        top_10_similar_chunks= [chunks[idx] for idx in top_indices]
        # top_10_similar_chunks = [expand_to_full_sentence(chunks, idx) for idx in top_indices]
        retrieved_context = "Answer is based on the following context:\n\n" + "\n\n".join(top_10_similar_chunks)

        # retrieved_context = ''.join(chunky for chunky in top_10_similar_chunks)
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            last_message = st.session_state.messages[-2]
        else:
            last_message = {"content": 'No responses given yet'}
            
        custom_prompt = f"""" You are a helpful assistant that based on retrieved documents and possibly previous responses returns an answer that fits with the question of the user.
                        Your role is to:
                        1. Answer questions by the user using the provided information.
                        2. Never generate information beyond what is retrieved from the document or provided earlier in the conversation.
                        3. Use information provided by the user or your own previous response.
                        The information to base your answer on:
                        - Retrieved Context: {retrieved_context}
                        - Helpful assistant's previous response : {last_message["content"]}
                        - User Question: {user_message}
                        Provide a constructive response that is to the point and as concise as possible. Base your answer only on the information provided.                        
                    """         
        response_text = get_model_response(custom_prompt, HF_client_LLM, model_name)
        st.session_state.messages.append({"role": "RetrievedChunks", "content": retrieved_context})
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        # Save to Supabase
        supabase_client.table("testEnvironment").insert({
            "session_id": st.session_state.session_id,
            "Question": user_message,
            "Answer": response_text,
            "LLM":model_name
        }).execute()

        st.rerun()
with col2:
    pdf_reader("airplaneNoImage.pdf")
