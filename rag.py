# ============================================================
# rag.py
#
# WHAT THIS FILE DOES:
# A complete RAG (Retrieval-Augmented Generation) system with a
# simple command-line chat interface (CLI).
#
# For every question the user types, it:
#   1. RETRIEVAL  -> finds the most relevant chunks (semantic search)
#   2. GENERATION -> sends those chunks to an LLM to write an answer
#   3. LATENCY    -> measures the total time (retrieval + generation)
#
# The user can keep asking questions until they type 'exit'.
# ============================================================

# ChromaDB is our vector database (stores chunks + embeddings)
import chromadb
# pandas reads the cleaned CSV data
import pandas
# time is used to measure how long the whole process takes
import time
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
    # Walk through the text in steps of 'size' characters
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ---------- STEP A: Set up the embedding model and the database ----------
print("Setting up the database...")

# Load the embedding model (turns each chunk into 384 numbers)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Open ChromaDB and create a collection to store our chunks
client_db = chromadb.Client()
collection = client_db.create_collection("my_articles")

# Read the cleaned dataset and use the first 50 articles
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)

# Split every article into chunks and gather them in one list
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

# Turn all chunks into embeddings and give each a unique ID
embeddings = model.encode(all_chunks)
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))

# Store the chunks, embeddings, and IDs in ChromaDB
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)
print("Database ready with", collection.count(), "chunks")
print()

# ---------- STEP B: Set up the LLM connection (OpenRouter) ----------
# base_url points to OpenRouter; api_key is our private key.
llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    api_key="PASTE_YOUR_KEY_HERE"
)

# ============================================================
# The main RAG function: takes a question, returns an answer.
# ============================================================
def answer_question(question):

    # ---------- STEP 1: RETRIEVAL ----------
    # Turn the question into numbers, then search for the top K chunks.
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=K)
    top_chunks = results["documents"][0]

    # ---------- STEP 2: BUILD THE CONTEXT ----------
    # Join all retrieved chunks into one block of text.
    # This is the "context" we will hand to the LLM.
    context = ""
    for chunk in top_chunks:
        context = context + chunk + "\n\n"

    # ---------- STEP 3: BUILD THE PROMPT ----------
    # system_prompt = tells the LLM its role and rules
    # user_prompt   = the context (chunks) + the actual question
    # This is "prompt engineering": clear role + context + instruction.
    system_prompt = "You are a helpful assistant. Answer the question using ONLY the context provided. If the answer is not in the context, say you don't know."
    user_prompt = "Context:\n" + context + "\nQuestion: " + question

    # ---------- STEP 4: GENERATION (with error handling) ----------
    # We wrap the API call in try/except so the program never crashes
    # if something goes wrong (rate limit, timeout, network, etc.).
    try:
        response = llm_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=30   # wait a maximum of 30 seconds, then give up
        )

        # MALFORMED RESPONSE CHECK:
        # Make sure the LLM actually returned some text.
        if response.choices and response.choices[0].message.content:
            answer = response.choices[0].message.content
            return answer
        else:
            return "Error: The LLM returned an empty response. Please try again."

    except Exception as e:
        # Any error is caught here so the program keeps running.
        # We look at the error text to give a helpful message.
        error_message = str(e)
        if "rate" in error_message.lower():
            # Too many requests too quickly
            return "Error: Too many requests (rate limit). Please wait and try again."
        elif "timeout" in error_message.lower():
            # The request took too long
            return "Error: The request took too long (timeout). Please try again."
        else:
            # Any other problem (network, bad key, etc.)
            return "Error: Something went wrong while contacting the LLM -> " + error_message

# ============================================================
# CLI: a simple chat loop so the user can ask questions
# ============================================================
print("=" * 50)
print("RAG Science Q&A System")
print("Ask any question about the articles.")
print("Type 'exit' to quit.")
print("=" * 50)
print()

# Keep asking questions until the user types 'exit'
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

    # Run the full RAG pipeline (retrieval + generation)
    answer = answer_question(question)

    # Stop the timer and calculate total time
    end_time = time.time()
    total_time = end_time - start_time

    # Show the answer and the latency
    print()
    print("Answer:")
    print(answer)
    print()
    print("(Latency:", round(total_time, 2), "seconds)")
    print("-" * 50)
    print()