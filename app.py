from dotenv import load_dotenv

load_dotenv()
import os
import pickle
import time

import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = ChatGroq(model="deepseek-r1-distill-llama-70b")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

file_name = "vector_index.pkl"
st.title("Covenant University Student Book Chatbot")
st.sidebar.title("Chat with me")

if not os.path.exists(file_name):
    process_url_clicked = st.sidebar.button("Process Handbook")


def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."], chunk_size=1000, chunk_overlap=400
    )

    docs = splitter.split_documents(pages)

    return docs


main_placeholder = st.empty()
progress_placeholder = st.empty()

if process_url_clicked:

    if os.path.exists(file_name):
        main_placeholder.text("Embedding already in memory. Ask your questions🧐")
    else:
        main_placeholder.text("Data Loading... , Started...✅✅✅.")
        url = (
            "https://www.covenantuniversity.edu.ng/downloads/Student-handbook-Feb-2020.pdf"
        )

        loader = PyPDFLoader(url)
        pages = loader.load_and_split()

        main_placeholder.text("Text Splitting... , Started...✅✅✅.. Please be Patient")
        docs = split_into_chunks(pages)

        data_embedding = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("Embedding Vector... , Started Building...✅✅✅")

        file_name = "vector_index.pkl"

        with open(file_name, "wb") as f:
            pickle.dump(data_embedding, f)

            main_placeholder.text("Embedding Saved...✅")


query = main_placeholder.text_input("Question :")
if query:
    progress_placeholder.text("Loading Embedding ..... Please wait")
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data_embedding = pickle.load(f)
        time.sleep(2)
        progress_placeholder.text("Searching Source data.....")
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=model, retriever=data_embedding.as_retriever()
        )
        progress_placeholder.text("Thinking .....")
        result = chain.invoke({"question": query}, return_only_output=True)
        progress_placeholder.text("DONE.....")
        st.write("Answer:")
        st.subheader(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)
                st.write(source)

    else:
        progress_placeholder.text("File not available, Please Process handbooks")
