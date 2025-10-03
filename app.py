import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

# --- UI Configuration ---
st.set_page_config(page_title="Chat with Your Book", page_icon="📚", layout="wide")
st.title("📚 Chat with Your Book")
st.markdown("---")

# --- App Description in Sidebar ---
with st.sidebar:
    st.header("About This Chatbot")
    st.markdown(
        "This chatbot is your personal assistant for the pre-loaded book. "
        "Ask any question about its content, and the AI will find the answer for you."
    )

# --- Define the path to your pre-uploaded book ---
PDF_PATH = "book.pdf"

@st.cache_resource
def create_qa_chain():
    """
    Sets up the entire RAG pipeline once and caches it.
    """
    # Load the OpenAI API key from Streamlit's secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found! Please add it to your Streamlit secrets.", icon="🚨")
        st.stop()
    
    # 1. Load the PDF from the hardcoded path
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # 3. Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 4. Create the FAISS Vector Store
    db = FAISS.from_documents(docs, embeddings)

    # 5. Set up the LLM using OpenAI's GPT model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

    # 6. Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

# --- Main Application Flow ---
st.header("Ask a Question About the Book")

# Create and display the chatbot interface
try:
    qa_chain = create_qa_chain()
    st.success("The book is loaded and the chatbot is ready.", icon="✅")

    # Chat interface
    user_question = st.text_input("What would you like to know?")

    if user_question:
        with st.spinner("Searching the book and generating an answer..."):
            result = qa_chain.invoke({"query": user_question})
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Show sources from the book"):
                st.write(result["source_documents"])
except Exception as e:
    st.error(f"An error occurred: {e}", icon="🔥")