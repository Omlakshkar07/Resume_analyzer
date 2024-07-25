import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = genai.Embeddings(api_key=GOOGLE_API_KEY)
    vectors = [embeddings.embed(text) for text in text_chunks]
    vector_store = FAISS(vectors)
    vector_store.save_local("faiss_index")

template = """
You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer. If you don't know the context, then don't answer the question.

context: \n{context}\n

question: \n{question}\n
Answer:"""

def get_conversational_chain():
    model = genai.Chat(api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def load_faiss_index(pickle_file):
    vectors = FAISS.load_local(pickle_file)
    return vectors

def user_input(user_question):
    embeddings = genai.Embeddings(api_key=GOOGLE_API_KEY)
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})

    st.write("Answer:", response["output_text"])

st.set_page_config(
    page_title="PDF Analyzer",
    page_icon=':books:',
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("Upload PDF")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing PDF"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("VectorDB Uploading Successful!!")

def main():
    st.title("LLM GenAi ChatBot")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("🖇Chat with PDF Analyzer 🗞")
    st.markdown("<hr>", unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="Ask a prompt")
    st.markdown("<hr>", unsafe_allow_html=True)

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
