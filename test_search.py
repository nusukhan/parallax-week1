# ============================================================
# test_search.py
# Tests retrieval performance by measuring how long ChromaDB
# takes to search for 10 different queries (latency test).
# ============================================================

# Import ChromaDB (our vector database)
import chromadb

# Import pandas to read the data
import pandas

# Import time to measure how long each search takes
import time

# Import the embedding model
from sentence_transformers import SentenceTransformer

# Import our chunking function
from chunk import chunk_text

# ---------- Set up the model and database ----------
# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client and a collection
client = chromadb.Client()
collection = client.create_collection("my_articles")

# ---------- Load data and make chunks ----------
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

# ---------- Generate embeddings and add to ChromaDB ----------
embeddings = model.encode(all_chunks)
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)

print("Database ready with", collection.count(), "chunks")
print()

# ============================================================
# Test retrieval latency for 10 different queries
# ============================================================

# A list of 10 different questions to test
queries = [
    "what is physics",
    "how do chemicals react",
    "what is biology",
    "explain gravity",
    "what is a computer",
    "how do stars form",
    "what is mathematics",
    "explain electricity",
    "what is energy",
    "how does the brain work"
]

print("Testing retrieval latency for 10 queries:")
print()

# Keep track of the total time across all queries
total_time = 0

# Go through each query one by one
for query in queries:
    # Convert the query into numbers (embedding)
    query_embedding = model.encode([query])

    # Start the timer, search, then stop the timer
    start = time.time()
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    end = time.time()

    # How long this one query took
    search_time = end - start
    total_time = total_time + search_time

    # Print the time for this query
    print("Query:", query, "-> Time:", round(search_time, 4), "seconds")

# ---------- Calculate and print the average ----------
average_time = total_time / len(queries)
print()
print("Total time for 10 queries:", round(total_time, 4), "seconds")
print("Average time per query:", round(average_time, 4), "seconds")