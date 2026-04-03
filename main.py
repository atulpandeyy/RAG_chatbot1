# ============================================================
#  RAG Chatbot Backend  –  LangChain + Flask
#  File: app.py
# ============================================================

import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
qa_chain = None       # the LangChain QA chain (built after PDF upload)
chat_history = []     # stores conversation turns for memory


# ── Step 1: Load PDF ──────────────────────────────────────────
def load_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    return pages


# ── Step 2: Split into chunks ─────────────────────────────────
def split_docs(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    return chunks


# ── Step 3: Build FAISS vector store ──────────────────────────
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── Step 4: Build conversational QA chain ────────────────────
def build_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Memory keeps track of previous questions and answers
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False
    )
    return chain


# ── Route: Serve the frontend HTML ───────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── Route: Upload PDF ─────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    global qa_chain, chat_history

    if "file" not in request.files:
        return jsonify({"error": "No file sent"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Build the pipeline
    pages = load_pdf(filepath)
    chunks = split_docs(pages)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_chain(vectorstore)
    chat_history = []   # reset chat on new upload

    return jsonify({
        "message": f"'{filename}' processed successfully!",
        "pages": len(pages),
        "chunks": len(chunks)
    })


# ── Route: Chat ───────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    global qa_chain

    if qa_chain is None:
        return jsonify({"error": "Please upload a PDF first!"}), 400

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Send a JSON body with a 'message' field"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Get answer from LangChain
    result = qa_chain.invoke({"question": user_message})
    answer = result["answer"]

    return jsonify({
        "question": user_message,
        "answer": answer
    })


if __name__ == "__main__":
    print("RAG Chatbot running at http://localhost:5000")
    app.run(debug=True, port=5000)
