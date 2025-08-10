## RAG Q&A Conversation with PDF Including Chat History
import sys

# Force Python to use pysqlite3 instead of the default sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
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
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Set up Streamlit

st.title("SAMPLE RAG Chatbot")
st.write("Upload your Pdf and chat with their content")

##Input the Groq API Key
api_key = os.getenv("GROQ_API_KEY")

## LLM model
llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

##Chat Interface
session_id = st.text_input("Session ID",value="default_session")

##Statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)

## Process Uploaded Files

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        tempPdf = f"./temp.pdf"
        with open(tempPdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)

        ## Split and create embeddings for the documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorStore = Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorStore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),

            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question Prompt

        system_prompt = (
            "You are an Assistant for the question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you dont know the answer, say that you"
            "dont know. Keep the answer concise and detailed."
            "concise.\n\n{context}"

        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("What do you wanna know:")
        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            # st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History:",session_history.messages)




