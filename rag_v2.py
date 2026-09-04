# ============================================================
# rag_v2.py
#
# WHAT THIS FILE DOES:
# A hallucination-resistant RAG system (Week 7). For every question:
#   1. RETRIEVAL          -> find the most relevant chunks (semantic search)
#   2. GENERATION         -> ask an LLM to answer using ONLY those chunks
#   3. HALLUCINATION CHECK -> compare the answer against the sources
#   4. STRUCTURED OUTPUT   -> return a JSON with answer, sources, score, latency
#
# It refuses (says "I don't know") when the context lacks the answer,
# and returns source citations so answers can be verified.
# ============================================================

# ChromaDB is our vector database (stores chunks + embeddings)
import chromadb
# pandas reads the cleaned CSV data
import pandas
# time is used to measure how long the whole process takes
import time
# json is used to build a clean, structured output
import json
# SentenceTransformer turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer
# OpenAI library is used to talk to the LLM through OpenRouter
from openai import OpenAI

# ---------- Best settings found during Week 5 evaluation ----------
CHUNK_SIZE = 700   # each chunk is 700 characters (gave best recall)
K = 5              # retrieve the top 5 chunks for each question

# ---------- Chunking function ----------
# Splits a long text into pieces of CHUNK_SIZE characters.
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ---------- Set up the embedding model and the database ----------
print("Setting up the database...")

model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.Client()
collection = client_db.create_collection("my_articles")

df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

embeddings = model.encode(all_chunks)
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)
print("Database ready with", collection.count(), "chunks")
print()

# ---------- Set up the LLM connection (OpenRouter) ----------
# api_key is our private key (kept as a placeholder for security).
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="PASTE_YOUR_KEY_HERE"
)

# ============================================================
# MAIN RAG FUNCTION
# Takes a question, returns the answer AND the source chunks used.
# ============================================================
def answer_question(question):

    # ---------- STEP 1: RETRIEVAL ----------
    # Turn the question into numbers, then search for the top K chunks.
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=K)
    top_chunks = results["documents"][0]

    # ---------- STEP 2: BUILD THE CONTEXT ----------
    # Join all retrieved chunks into one block of text for the LLM.
    context = ""
    for chunk in top_chunks:
        context = context + chunk + "\n\n"

    # ---------- STEP 3: BUILD THE PROMPT (anti-hallucination) ----------
    # Strong instructions: use ONLY the context, no outside knowledge,
    # and say "I don't know" if the answer is missing.
    system_prompt = (
        "You are a helpful assistant for a science Q&A system. "
        "Answer the question using ONLY the information in the context below. "
        "Do NOT use any outside knowledge. "
        "If the answer is not in the context, reply with exactly: "
        "'I don't know based on the provided documents.' "
        "Do not make up any information."
    )
    user_prompt = "Context:\n" + context + "\nQuestion: " + question

    # ---------- STEP 4: GENERATION (with error handling) ----------
    # try/except so the program never crashes on an API problem.
    try:
        response = llm_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=30   # wait a maximum of 30 seconds, then give up
        )

        # Malformed check: make sure the LLM returned some text
        if response.choices and response.choices[0].message.content:
            answer = response.choices[0].message.content
            return answer, top_chunks
        else:
            return "Error: The LLM returned an empty response. Please try again.", top_chunks

    except Exception as e:
        # Catch any error and return a helpful message (no crash)
        error_message = str(e)
        if "rate" in error_message.lower():
            return "Error: Too many requests (rate limit). Please wait and try again.", top_chunks
        elif "timeout" in error_message.lower():
            return "Error: The request took too long (timeout). Please try again.", top_chunks
        else:
            return "Error: Something went wrong while contacting the LLM -> " + error_message, top_chunks

# ============================================================
# HALLUCINATION CHECK
# Compares the answer against the source chunks. If most of the
# answer's words are found in the sources, it is "supported";
# otherwise it is flagged as possibly hallucinated.
# ============================================================
def check_hallucination(answer, source_chunks):
    # Join all source chunks into one big text (lowercased)
    source_text = ""
    for chunk in source_chunks:
        source_text = source_text + chunk.lower() + " "

    # Break the answer into words (lowercased)
    answer_words = answer.lower().split()

    # Count how many "real" answer words appear in the source text
    supported = 0
    total_checked = 0
    for word in answer_words:
        # Only check longer words (skip short ones like "a", "is", "the")
        if len(word) > 4:
            total_checked = total_checked + 1
            if word in source_text:
                supported = supported + 1

    # Avoid divide by zero
    if total_checked == 0:
        return "Unknown", 0

    # Fraction of the answer supported by the sources
    support_score = supported / total_checked

    if support_score >= 0.5:
        status = "Supported (likely grounded in sources)"
    else:
        status = "WARNING: possibly hallucinated (not well supported)"

    return status, round(support_score, 2)

# ============================================================
# CLI: a simple chat loop so the user can ask questions
# ============================================================
print("=" * 50)
print("RAG Science Q&A System (Hallucination-Resistant)")
print("Ask any question about the articles.")
print("Type 'exit' to quit.")
print("=" * 50)
print()

while True:
    # Get a question from the user
    question = input("Your question: ")

    # If the user wants to quit, stop the loop
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Skip empty questions (edge case)
    if question.strip() == "":
        print("Please type a question.")
        print()
        continue

    # Start the timer (to measure end-to-end latency)
    start_time = time.time()

    # Run the full RAG pipeline — returns answer AND source chunks
    answer, source_chunks = answer_question(question)

    # Stop the timer and calculate total time
    end_time = time.time()
    total_time = end_time - start_time

    # If the LLM refused (said it doesn't know), that's CORRECT, not a hallucination
    if "i don't know" in answer.lower() or "cannot answer" in answer.lower():
        status = "Refused (correctly said it doesn't know)"
        score = 1.0
    else:
        # Otherwise, run the hallucination check
        status, score = check_hallucination(answer, source_chunks)

    # ---------- Build a structured JSON output ----------
    # Take a short preview of each source chunk (first 100 characters)
    source_previews = []
    for chunk in source_chunks:
        source_previews.append(chunk[:100])

    # Build the structured result as a dictionary
    result = {
        "question": question,
        "answer": answer,
        "hallucination_check": status,
        "support_score": score,
        "sources": source_previews,
        "latency_seconds": round(total_time, 2)
    }

    # Print the result as nicely formatted JSON
    print()
    print("Structured JSON output:")
    print(json.dumps(result, indent=2))
    print("-" * 50)
    print()