import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import chromadb
from chromadb.config import Settings
import chromadb.utils
from sentence_transformers import SentenceTransformer
import hashlib
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# --- Load Hugging Face API ---
api_key = os.getenv("huggingface_api_key")
client = InferenceClient(api_key=api_key)
model = "openai/gpt-oss-20b"

# --- Load embedding model and ChromaDB ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Try to use local persistent DB, fallback to in-memory
try:
    chroma_client = chromadb.PersistentClient(path="./faq_chromadb")
except Exception:
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None))

collection = chroma_client.get_or_create_collection("faq_collection")

# --- Embedding cache (in-memory) ---
embedding_cache = {}

# --- System prompt ---
base_prompt = "You are a helpful customer service bot that answers based on the provided context."

# --- Session storage for query counts ---
session_queries = {}

# --- Helper: Cache embeddings ---
def get_cached_embedding(text):
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    emb = embedding_model.encode([text])[0].tolist()
    embedding_cache[key] = emb
    return emb

# --- RAG-powered chat function ---
def chat(user_query: str):
    try:
        query_emb = get_cached_embedding(user_query)
        results = collection.query(query_embeddings=[query_emb], n_results=2)
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."

        rag_prompt = (
            "Context:\n"
            + context
            + "\n\nQuestion:\n"
            + user_query
            + "\n\nAnswer based on the above context. "
              "If not sure, politely say you don't have enough information."
        )

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": rag_prompt},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"[Error] Unexpected issue: {e}"

# --- FastAPI setup ---
app = FastAPI(title="HelpGenie Chatbot API")

# --- Enable CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve static files (CSS/JS) ---
#app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Serve index.html at root ---
#@app.get("/")
#def serve_index():
#    return FileResponse(Path("static/index.html"))

# --- Request schema ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

MAX_QUERIES = 5  # max queries per session

# --- API endpoint ---
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query.strip()

    # Check query limit
    count = session_queries.get(session_id, 0)
    if count >= MAX_QUERIES:
        raise HTTPException(
            status_code=403,
            detail="You have reached the maximum number of queries for this session."
        )

    # Update query count
    session_queries[session_id] = count + 1

    # Get bot response
    answer = chat(user_query)

    return {
        "answer": answer,
        "queries_used": session_queries[session_id],
        "queries_remaining": MAX_QUERIES - session_queries[session_id]
    }
