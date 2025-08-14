import pysqlite3
import sys

# The fix must be here, at the top.
sys.modules["sqlite3"] = sys.modules["pysqlite3"]


import os
import streamlit as st
import torch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit
st.title("SAMPLE RAG Chatbot")
st.write("Upload your PDF and chat with its content")

# Statefully manage chat history and RAG chain
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None

# Caching LLM and embeddings to prevent re-initialization on every rerun
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

@st.cache_resource
def get_embeddings():
    # Check for GPU and set device to prevent 'meta tensor' error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

llm = get_llm()
embeddings = get_embeddings()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Returns a ChatMessageHistory for the given session ID."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# UI for session ID
session_id = st.text_input("Session ID", value="default_session")

# UI for file uploader
uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", accept_multiple_files=True)

# Process uploaded files and set up the RAG chain
if uploaded_files and st.session_state.conversational_rag_chain is None:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf_path = f"./temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(temp_pdf_path)  # Clean up the temporary file

    # Split and create embeddings for all documents at once
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    # Create the vector store
    vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorStore.as_retriever()

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question prompt
    system_prompt = (
        "You are an Assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the"
        "answer concise."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create the conversational chain and store in session state
    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    st.success("PDFs processed and chatbot is ready to use!")

# Chat interface
if st.session_state.conversational_rag_chain:
    user_input = st.text_input("What do you want to know:")
    if user_input:
        response = st.session_state.conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.success(f"Assistant: {response['answer']}")
        session_history = get_session_history(session_id)
        st.write("Chat History:", session_history.messages)
