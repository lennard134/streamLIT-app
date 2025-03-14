import streamlit as st
from together import Together
from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf
from langchain.text_splitter import CharacterTextSplitter
import re
from sentence_transformers import SentenceTransformer
import openai
import torch
from supabase import create_client, Client

import hashlib
import time
import random

def pdf_to_text(pdf_path):
    """Extracts text from a PDF and returns it as a single long string."""
    doc = pymupdf.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces/newlines

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


def document_parsing(file_path, chunk_size):
    """
    Document parser, processes uploaded document and splits text into chunks for a given chunksize.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_size/10, length_function=len)
    return text_splitter.split_documents(pages)

def save_chat_to_db():
    st.write(f"Implement code that writes data to DB")

# Function for evaluation (Can be expanded)
def evaluate_chat():
    st.write("Evaluating session...")
    num_interactions = len(st.session_state.messages) // 2  # Assuming user-bot pairs
    st.write(f"Total interactions: {num_interactions}")

#embed documents
# Initialization
if 'session_id' not in st.session_state:
    session_data = f"{time.time()}_{random.randint(0,int(1e6))}".encode()
    st.session_state['session_id'] = hashlib.sha256(session_data).hexdigest()[:16]
if 'embeddings' not in st.session_state:  
    # chunk_size = 512
    # embedding = "nomic-embed-text"
    # file_path = '../Data/content/Horowitz.pdf'
    # docs = document_parsing(file_path, chunk_size=chunk_size)
    # local_embeddings = OllamaEmbeddings(model=embedding)
    # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size/10)
    # texts = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=texts, embedding=local_embeddings)
    # retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    # print(type(retriever))
    embed_name = 'all-MiniLM-L6-v2' 
    # model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = SentenceTransformer(embed_name)
    pdf_path = 'schoolgids.pdf'
    chunk_size = 1024
    data_string = pdf_to_text(pdf_path=pdf_path)
    ## Chunks is list of strings
    chunks = split_into_chunks(data_string, chunk_size=chunk_size)
    embeddings = model.encode(chunks)
    st.session_state['embeddings'] = embeddings
    st.session_state['model'] = model
    st.session_state['chunks'] = chunks

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title & Intro
st.title("Chat with your assistant 🤖")
# Close and Evaluate Button
if st.button("Close and Evaluate"):
    pass
    #TODO implement evaluation question
    # save_chat_to_db()
    # evaluate_chat()
    # st.session_state.messages = []  # Clear chat after closing
    # st.rerun()
# A murder has been committed in the quiet neighborhood of Riverview Close. Giles Kenworthy has been found dead—killed by a crossbow bolt to the neck. His death seems suspicious, but the case is far from simple. Some say it was a crime of passion, others believe it was carefully orchestrated. The question is: who killed him and why?
# Your job is to investigate the case, uncover clues, and piece together the truth. But be careful—not everything is as it seems.
# What we know so far is that the Victim is Giles Kenworthy and he was killed with a crossbow shot to the neck!
# Can you uncover what happened, be as clear as possible when asking the questions? Good luck! 

st.write("""Chat hier met een document van CBS de Citer, wil jij weten wat de vrijwillige ouderbijdrage is, vraag het gerust!""")

# User Input
token = st.secrets["TOGETHER_API_TOKEN"]
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

client = openai.OpenAI(
  api_key=token,
  base_url="https://api.together.xyz/v1",
)
supabase_client: Client = create_client(url, key)


# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_message = st.chat_input("Ask your question here")
llm_response = ''
if user_message:
    # Append user message to chat history

    question_embedded = st.session_state.model.encode(user_message)
    similarities = st.session_state.model.similarity(st.session_state.embeddings, question_embedded)
    # Flatten the tensor (if it's a column vector)
    tensor_values = similarities.view(-1)

    # Get top 5 values and their indices
    top5_values, top5_indices = torch.topk(tensor_values, k=5)
    retrieved_context = ''
    for idx in top5_indices:
        retrieved_context += st.session_state.chunks[idx]
    print(retrieved_context)

    st.session_state.messages.append({"role": "user", "content": user_message})
    
    if "messages" in st.session_state:  
        last_message = st.session_state.messages[-1]
        print(f'Last message: {last_message}')
    else:
        last_message = ''
                    # You are assisting a detective in solving a murder commited on Giles Kenworthy. 
                    # You have access to a document that contains all the detail.
                    # The detective wants to know who committed the murder and what the motive was.
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
    # Send request to LLM API
    
    result = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "assistant", "content": custom_prompt}],
            )

    llm_response =result.choices[0].message.content
    # Append LLM response to chat history
    response = (
        supabase_client.table("testEnvironment")
        .insert({"session_id": st.session_state.session_id, "Question": user_message, "Answer": llm_response})
        .execute()  
    )
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # Clear user input
    st.rerun()

