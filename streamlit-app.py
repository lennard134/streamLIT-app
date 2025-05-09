
import re
import openai
import hashlib
import time
import os
import random
import requests
import tempfile
import pickle 
import base64
import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine

from supabase import create_client, Client
from streamlit_pdf_reader import pdf_reader
from huggingface_hub import InferenceClient


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

# Session ID
if 'session_id' not in st.session_state:
    session_data = f"{time.time()}_{random.randint(0,int(1e6))}".encode()
    st.session_state['session_id'] = hashlib.sha256(session_data).hexdigest()[:16]

# Load & cache resources
chunks, embeddings = get_chunks_and_embeddings()

def get_embedding_with_retry(user_message, HF_client, max_retries=2, wait_time=1):
    retries = 0
    while retries < max_retries:
        try:
            question_embed = HF_client.feature_extraction(
                user_message,
                model="intfloat/multilingual-e5-large-instruct",
            )
            if question_embed is not None:
                return question_embed
            else:
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
    out = 'Bad Request: Please try again'
    return out
    
def get_model_response(user_message, HF_client, model_name, max_retries=2, wait_time=1):
    retries = 0
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
                wait_time = wait_time * 2  # Exponentially increase wait time
                print(f"Retrying... {retries}/{max_retries}")
                time.sleep(wait_time)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed due to error: {e}")
            retries += 1
            wait_time = wait_time * 2  # Exponentially increase wait time
            print(f"Retrying... {retries}/{max_retries}")
            time.sleep(wait_time)
    
    out = 'Bad Request: Please try again'
    return out

# UI

with st.sidebar:
    user_input = st.text_input("Please insert your unique identifier", "tr...")

    # Define the regular expression pattern for "tr" followed by exactly three digits
    pattern = r"^tr\d{3}$"
    
    if not re.match(pattern, user_input):
        st.warning('Retrieve your ID from the qualtrics environment and insert here', icon="⚠️")
        st.session_state["MODEL_CHOSEN"] = False
    else:
        user_input = user_input.replace('tr', '')
        user_input = int(user_input)
        st.session_state["MODEL_CHOSEN"] = True
        if user_input < 100:
            model_name = "google/gemma-2-2b-it"#"meta-llama/Llama-3.2-1B-Instruct"
        elif 100 <= user_input < 500:
            model_name = "google/gemma-2-9b-it"#"meta-llama/Llama-3.2-3B-Instruct"
        else:
            model_name = "google/gemma-2-27b-it"#"meta-llama/Llama-3.2-3B-Instruct"
            
    
if st.session_state["MODEL_CHOSEN"] == True:
    with col1:
        st.header("💬 Assistant")
    
        # Secrets
        # token = st.secrets["TOGETHER_API_TOKEN"]
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        HF_TOKEN = st.secrets["HF_API_TOKEN"]
        
        # client = openai.OpenAI(api_key=token, base_url="https://api.together.xyz/v1")

        supabase_client: Client = create_client(url, key)
        HF_client_LLM = InferenceClient(
            provider="nebius",
            api_key=HF_TOKEN,
        )

        HF_client_Feature = InferenceClient(
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
            question_embed = get_embedding_with_retry(user_message, HF_client_Feature)
            print(f'question embedded:{question_embed}')
            similarities = []
            for chunk_embedding in embeddings:
                similarity = 1 - cosine(question_embed, chunk_embedding)
                similarities.append(similarity)

            top_indices = np.argsort(similarities)[::-1][:5]  # Indices of the top 10 similar chunks
            
            # Retrieve the top 10 most similar chunks based on the indices
            top_10_similar_chunks= [chunks[idx] for idx in top_indices]
            # top_10_similar_chunks = [expand_to_full_sentence(chunks, idx) for idx in top_indices]
            retrieved_context = "Answer based on the following context:\n" + "\n\n".join(top_10_similar_chunks)
    
            # retrieved_context = ''.join(chunky for chunky in top_10_similar_chunks)
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
else:
    with col1:
        st.write("Please fill in your ID on the sidebar")
    with col2:
        st.write("Please fill in your ID on the sidebar")
        
