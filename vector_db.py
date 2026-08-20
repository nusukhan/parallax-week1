# ============================================================
# vector_db.py
#
# WHAT THIS FILE DOES:
# Sets up ChromaDB, stores our chunks + embeddings, and runs
# semantic search to find the most relevant chunks for a query.
#
# This version uses the BEST settings found during the Week 5
# evaluation experiments:
#   - Chunk size = 700 characters (gave the best recall)
#   - K = 5 (returning the top 5 chunks gave the best recall)
# ============================================================

# ChromaDB is our vector database (stores chunks + embeddings)
import chromadb
# pandas reads the cleaned CSV data
import pandas
# SentenceTransformer turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer

# ---------- Best settings from Week 5 evaluation ----------
# These values were chosen because they gave the best
# Precision/Recall results during evaluation.
CHUNK_SIZE = 700   # each chunk is 700 characters long
K = 5              # search returns the top 5 chunks

# ---------- Chunking function (uses the best chunk size) ----------
# Splits a long text into pieces of CHUNK_SIZE characters.
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    # Walk through the text in steps of 'size' characters
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ---------- Load the model and set up ChromaDB ----------
# The embedding model (turns each chunk into 384 numbers)
model = SentenceTransformer("all-MiniLM-L6-v2")
# Open ChromaDB and create a collection to store our data
client = chromadb.Client()
collection = client.create_collection("my_articles")

# ---------- Load the data and split into chunks ----------
# Read the cleaned dataset and use the first 50 articles
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)

# Split every article into chunks and gather them in one list
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

print("Total chunks:", len(all_chunks))

# ---------- Create embeddings and store in ChromaDB ----------
# Turn all chunks into embeddings (numbers)
embeddings = model.encode(all_chunks)

# Give each chunk a unique ID: chunk_0, chunk_1, ...
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))

# Store the chunks, their embeddings, and their IDs in ChromaDB
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)

print("Added", len(all_chunks), "chunks to ChromaDB!")
print()

# ---------- Semantic search using the best K ----------
# The question we want to search for
query = "what is physics"

# Turn the question into numbers (embedding)
query_embedding = model.encode([query])

# Search ChromaDB for the top K most similar chunks
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=K)

# Print the query and the matching chunks
print("Query:", query)
print("Top", K, "matching chunks:")
for doc in results["documents"][0]:
    print("-", doc[:100])