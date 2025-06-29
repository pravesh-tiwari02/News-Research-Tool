import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("🗞️ News Research Tool")
st.sidebar.title("📰 Enter News Article URLs")

# Input URLs
urls = [st.sidebar.text_input(f"URL {i+1}", "") for i in range(3)]
urls = [url.strip() for url in urls if url.strip()]
process_url_clicked = st.sidebar.button("🔍 Process URLs")

file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.6
)

# Process URLs
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔄 Loading data from URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100
        )
        main_placeholder.text("✂️ Splitting text...")
        docs = text_splitter.split_documents(data)

        main_placeholder.text("📦 Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

        main_placeholder.success("✅ Processing complete. Ask your questions below.")

# Ask question
query = st.text_input("🔎 Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("Please process URLs first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query})

        st.header("📖 Answer")
        st.write(result.get("answer", "No answer found."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("📚 Sources:")
            for src in sources.split("\n"):
                if src.strip():
                    st.write(src)
