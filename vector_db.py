# ============================================================
# vector_db.py
# Sets up ChromaDB, ingests chunks + embeddings, and runs
# a basic semantic search to retrieve the top-K chunks.
# ============================================================

# Import ChromaDB (our vector database)
import chromadb

# Import pandas to read the cleaned data
import pandas

# Import the model that turns text into numbers (embeddings)
from sentence_transformers import SentenceTransformer

# Import our chunking function from chunk.py
from chunk import chunk_text

# ---------- Step 1: Load the embedding model ----------
# This model turns each chunk into 384 numbers
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Step 2: Set up ChromaDB ----------
# Create a ChromaDB client (opens the database)
client = chromadb.Client()

# Create a collection (a place to store chunks and embeddings)
collection = client.create_collection("my_articles")

# ---------- Step 3: Load and chunk the data ----------
# Read the cleaned dataset
df = pandas.read_csv("cleaned_articles.csv")

# Take a subset of 50 articles (to keep it fast)
subset = df.head(50)

# Break every article into chunks and collect them in one list
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

print("Total chunks:", len(all_chunks))

# ---------- Step 4: Generate embeddings ----------
# Turn all chunks into embeddings (numbers)
embeddings = model.encode(all_chunks)

# ---------- Step 5: Create IDs for each chunk ----------
# Every chunk needs a unique ID (chunk_0, chunk_1, ...)
ids = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))

# ---------- Step 6: Add everything to ChromaDB ----------
# Store the chunks (documents), their embeddings, and IDs
collection.add(
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    ids=ids
)

print("Added", len(all_chunks), "chunks to ChromaDB!")
print("Collection count:", collection.count())

# ============================================================
# Semantic Search (retrieve the top-K chunks for a query)
# ============================================================

# The question we want to search for
query = "what is physics"

# Convert the question into numbers (embedding)
query_embedding = model.encode([query])

# Search ChromaDB for the top 3 most similar chunks
# n_results=3 means we want the top-K where K = 3
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

# Print the query and the matching chunks
print()
print("Query:", query)
print("Top 3 matching chunks:")
for doc in results["documents"][0]:
    print("-", doc[:100])