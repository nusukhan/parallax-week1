# ============================================================
# api.py — Week 10: FastAPI wrapper for the RAG system
#
# WHAT THIS FILE DOES:
# Turns the RAG system into a web API (a service others can call).
# It exposes three endpoints:
#   /query    -> ask a question (returns RAG answer + hallucination check)
#   /metadata -> get info about the corpus (chunks, model, settings)
#   /health   -> check the system is running
# It also logs every request to api_log.txt and returns proper
# HTTP error codes (400 / 422 / 500) when something is wrong.
# ============================================================

# time -> measure how long a request takes (latency)
import time
# json -> (available for structured data if needed)
import json
# logging -> write each request to a log file
import logging

# pandas -> read the CSV data
import pandas
# chromadb -> our vector database
import chromadb
# SentenceTransformer -> turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer
# OpenAI -> used to talk to the LLM through OpenRouter
from openai import OpenAI

# FastAPI -> the web framework; HTTPException -> to send error codes
from fastapi import FastAPI, HTTPException
# BaseModel -> defines the shape of the incoming request (auto-validation)
from pydantic import BaseModel

# ---------- Settings ----------
CHUNK_SIZE = 700   # each chunk is 700 characters
K = 5              # retrieve the top 5 chunks per query

# ---------- Logging setup ----------
# Every request will be written to api_log.txt with a timestamp.
logging.basicConfig(
    filename="api_log.txt",       # the log file name
    level=logging.INFO,           # log INFO-level messages and above
    format="%(asctime)s | %(message)s"   # timestamp | message
)

# ---------- Chunking function ----------
# Splits a long text into pieces of CHUNK_SIZE characters.
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ============================================================
# Build the RAG database ONCE when the server starts
# (so every request can reuse it — fast).
# ============================================================
print("Setting up the RAG system...")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create the ChromaDB collection
client_db = chromadb.Client()
collection = client_db.create_collection("api_articles")

# Read data and build chunks (first 50 articles to stay fast)
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)
all_chunks = []
for content in subset["content"]:
    all_chunks = all_chunks + chunk_text(str(content))

# Turn chunks into embeddings and give each an ID
embeddings = model.encode(all_chunks)
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))

# Store everything in ChromaDB
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)
print("Database ready with", collection.count(), "chunks")

# ---------- LLM client (OpenRouter) ----------
# base_url = OpenRouter address; api_key = our private key
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="PASTE_YOUR_KEY_HERE"
)

# ============================================================
# Hallucination check: compare the answer against the sources.
# Returns a status and a support score (0 to 1).
# ============================================================
def check_hallucination(answer, source_chunks):
    # Join all source chunks into one lowercased text
    source_text = ""
    for chunk in source_chunks:
        source_text = source_text + chunk.lower() + " "

    # Break the answer into words
    answer_words = answer.lower().split()

    # Count how many long words appear in the sources
    supported = 0
    total = 0
    for word in answer_words:
        if len(word) > 4:
            total = total + 1
            if word in source_text:
                supported = supported + 1

    # Avoid divide by zero
    if total == 0:
        return "Unknown", 0

    # Fraction of the answer supported by the sources
    score = supported / total
    if score >= 0.5:
        return "Supported", round(score, 2)
    return "WARNING: possibly hallucinated", round(score, 2)

# ============================================================
# The RAG function: retrieve chunks + generate an answer.
# Returns the answer AND the source chunks used.
# ============================================================
def answer_question(question):
    # Retrieve: search for the top K chunks
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=K)
    top_chunks = results["documents"][0]

    # Build the context from the retrieved chunks
    context = ""
    for chunk in top_chunks:
        context = context + chunk + "\n\n"

    # Prompt: use only the context, else say "I don't know"
    system_prompt = (
        "You are a helpful assistant. Answer using ONLY the context. "
        "If the answer is not in the context, say: "
        "'I don't know based on the provided documents.'"
    )
    user_prompt = "Context:\n" + context + "\nQuestion: " + question

    # Generate: send to the LLM and get the answer
    response = llm_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        timeout=30
    )
    answer = response.choices[0].message.content
    return answer, top_chunks

# ============================================================
# FastAPI app — this is the web service
# ============================================================
# Creating the app object; title/description show on the /docs page.
app = FastAPI(title="RAG API", description="Week 10 RAG FastAPI service")

# The shape of the /query request body.
# If the user sends the wrong shape, FastAPI auto-returns 422.
class QueryRequest(BaseModel):
    question: str

# ---------- /health : check the system is running ----------
# A GET endpoint that simply confirms the service is alive.
@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_db": collection.count()}

# ---------- /metadata : info about the corpus ----------
# A GET endpoint returning details about the data and settings.
@app.get("/metadata")
def metadata():
    return {
        "total_articles_used": len(subset),
        "total_chunks": collection.count(),
        "chunk_size": CHUNK_SIZE,
        "top_k": K,
        "embedding_model": "all-MiniLM-L6-v2"
    }

# ---------- /query : ask a question ----------
# A POST endpoint that takes a question and returns the RAG answer.
@app.post("/query")
def query(request: QueryRequest):
    question = request.question

    # 400 error: the question is empty or just spaces (malformed)
    if question is None or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Start timing the request
    start_time = time.time()
    try:
        # Run the RAG pipeline
        answer, source_chunks = answer_question(question)
        # Run the hallucination check
        status, score = check_hallucination(answer, source_chunks)
        # Calculate total latency
        latency = round(time.time() - start_time, 2)

        # Log this request: query, number of chunks, latency, answer preview
        logging.info(
            "QUERY=%s | CHUNKS=%d | LATENCY=%s | ANSWER=%s",
            question, len(source_chunks), latency, answer[:100]
        )

        # Return the structured JSON response
        return {
            "question": question,
            "answer": answer,
            "hallucination_check": status,
            "support_score": score,
            "sources": [c[:100] for c in source_chunks],
            "latency_seconds": latency
        }

    except Exception as e:
        # 500 error: something went wrong (e.g. LLM/API failure)
        logging.info("ERROR for QUERY=%s | %s", question, str(e))
        raise HTTPException(status_code=500, detail="Internal error: " + str(e))