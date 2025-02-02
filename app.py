<<<<<<< HEAD
=======
from dotenv import load_dotenv

load_dotenv()
>>>>>>> d5d242418536da8854d38fd2ec0c9428f9ea0fd0
import os
import pickle
import time

<<<<<<< HEAD
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
=======
import streamlit as st
>>>>>>> d5d242418536da8854d38fd2ec0c9428f9ea0fd0
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
<<<<<<< HEAD
from telegram import Bot, Update
from telegram.ext import Dispatcher

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Telegram Bot
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TOKEN)

# This will store the dispatcher to handle updates
dispatcher = Dispatcher(bot, None, workers=0)

# Model and embeddings initialization
model = ChatGroq(model="deepseek-r1-distill-llama-70b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
file_name = "vector_index.pkl"
=======

model = ChatGroq(model="deepseek-r1-distill-llama-70b")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

file_name = "vector_index.pkl"
st.title("Covenant University Student Book Chatbot")
st.sidebar.title("Chat with me")


process_url_clicked = st.sidebar.button("Process Handbook")

>>>>>>> d5d242418536da8854d38fd2ec0c9428f9ea0fd0

def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."], chunk_size=1000, chunk_overlap=400
    )
<<<<<<< HEAD
    docs = splitter.split_documents(pages)
    return docs

# Command handler for the /start command
async def start(update: Update, context):
    await update.message.reply_text('Welcome to the Covenant University Student Book Chatbot! Use /process to load the handbook.')

async def process_handbook(update: Update, context):
    await update.message.reply_text("Data Loading... , Started...✅✅✅.")
    url = "https://www.covenantuniversity.edu.ng/downloads/Student-handbook-Feb-2020.pdf"
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    await update.message.reply_text("Text Splitting... , Started...✅✅✅.. Please be Patient")
    docs = split_into_chunks(pages)
    data_embedding = FAISS.from_documents(docs, embeddings)
    await update.message.reply_text("Embedding Vector... , Started Building...✅✅✅")
    with open(file_name, "wb") as f:
        pickle.dump(data_embedding, f)
    await update.message.reply_text("Embedding Saved...✅")

async def handle_message(update: Update, context):
    query = update.message.text
    if query:
        await update.message.reply_text("Loading Embedding ..... Please wait")
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                data_embedding = pickle.load(f)
            time.sleep(2)
            await update.message.reply_text("Searching Source data.....")
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=model, retriever=data_embedding.as_retriever()
            )
            await update.message.reply_text("Thinking .....")
            result = chain.invoke({"question": query}, return_only_output=True)
            await update.message.reply_text("DONE.....")
            await update.message.reply_text(f"Answer: {result['answer']}")
            sources = result.get("sources", "")
            if sources:
                await update.message.reply_text("Sources:")
                source_list = sources.split("\n")
                for source in source_list:
                    await update.message.reply_text(source)
        else:
            await update.message.reply_text("File not available, Please Process handbooks")

# Set up the webhook endpoint
@app.post(f"/{TOKEN}")
async def webhook(request: Request):
    update = Update(**(await request.json()))
    dispatcher.process_update(update)
    return JSONResponse(content={"status": "ok"})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    uvicorn.run(app, host="0.0.0.0", port=8000)
=======

    docs = splitter.split_documents(pages)

    return docs


main_placeholder = st.empty()
progress_placeholder = st.empty()

if process_url_clicked:

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
# >>>>>>> d5d242418536da8854d38fd2ec0c9428f9ea0fd0
