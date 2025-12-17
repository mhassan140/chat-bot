from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2, api_key=api_key)
st.set_page_config(page_title="Gemini PDF Chatbot")
st.title("Mohsin Rag System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    st.success(f"PDF loaded successfully! Total chunks: {len(chunks)}")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    vectorstore.persist()

query = st.text_input("Ask a question from the PDF")

if query:
    top_chunks = vectorstore.similarity_search(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc in top_chunks])
    prompt = f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {query}"
    answer = llm.invoke(prompt)
    st.subheader("Answer")
    st.write(answer.content)








