# ============================================================
# evaluate.py
#
# WHAT THIS FILE DOES:
# Measures how good our semantic search is, using two metrics:
#   - Precision@K : out of the K chunks returned, how many were correct?
#   - Recall@K    : did we find at least one correct chunk?
#
# It also EXPERIMENTS with different settings (hyperparameters):
#   - Chunk size : 300, 500, 700 characters
#   - K value    : 3 or 5 top chunks
# This helps us find which combination gives the best results.
# ============================================================

# ChromaDB is our vector database
import chromadb
# pandas reads the cleaned CSV data
import pandas
# SentenceTransformer turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer


# ---------- A flexible chunking function ----------
# Unlike chunk.py (fixed at 500), this takes the chunk size as
# an input, so we can experiment with different sizes.
def chunk_text_size(text, size):
    chunks = []
    # Walk through the text in steps of 'size' characters
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks


# ---------- Load the model and data once ----------
# The embedding model (turns each chunk into 384 numbers)
model = SentenceTransformer("all-MiniLM-L6-v2")
# ChromaDB client (we will make one collection per chunk size)
client = chromadb.Client()

# Read the cleaned dataset and use the first 50 articles
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)


# ---------- Test set: 20 queries with known ground truth ----------
# For each query, "ground_truth" is the keyword a correct chunk
# should contain. Example: "what is gravity" -> "gravity".
test_set = [
    {"query": "what is physics", "ground_truth": "physics"},
    {"query": "what is biology", "ground_truth": "biology"},
    {"query": "what is chemistry", "ground_truth": "chemistry"},
    {"query": "what is gravity", "ground_truth": "gravity"},
    {"query": "how do stars form", "ground_truth": "star"},
    {"query": "what is energy", "ground_truth": "energy"},
    {"query": "what is a cell", "ground_truth": "cell"},
    {"query": "what is an atom", "ground_truth": "atom"},
    {"query": "what is electricity", "ground_truth": "electric"},
    {"query": "what is light", "ground_truth": "light"},
    {"query": "what is a molecule", "ground_truth": "molecule"},
    {"query": "what is evolution", "ground_truth": "evolution"},
    {"query": "what is a planet", "ground_truth": "planet"},
    {"query": "what is force", "ground_truth": "force"},
    {"query": "what is mathematics", "ground_truth": "mathematics"},
    {"query": "what is a computer", "ground_truth": "computer"},
    {"query": "what is dna", "ground_truth": "dna"},
    {"query": "what is heat", "ground_truth": "heat"},
    {"query": "what is a wave", "ground_truth": "wave"},
    {"query": "what is matter", "ground_truth": "matter"},
]


# ---------- Function: build a database for a given chunk size ----------
# Chunks all 50 articles at the given size, creates embeddings,
# and stores everything in a fresh ChromaDB collection.
def build_collection(chunk_size):
    # A separate collection name for each chunk size
    collection = client.create_collection("test_" + str(chunk_size))

    # Chunk every article using this chunk size
    all_chunks = []
    for content in subset["content"]:
        chunks = chunk_text_size(str(content), chunk_size)
        all_chunks = all_chunks + chunks

    # Turn chunks into embeddings and give each a unique ID
    embeddings = model.encode(all_chunks)
    ids = []
    for i in range(len(all_chunks)):
        ids.append("chunk_" + str(i))

    # Store chunks, embeddings, and IDs in ChromaDB
    collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)
    return collection, len(all_chunks)


# ---------- Function: evaluate one collection at a given K ----------
# Runs all 20 queries and returns the average precision and recall.
def evaluate_at_k(collection, k):
    total_precision = 0
    total_recall = 0

    # Test every query
    for item in test_set:
        query = item["query"]
        truth = item["ground_truth"]

        # Turn the query into numbers and search for top k chunks
        query_embedding = model.encode([query])
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
        top_chunks = results["documents"][0]

        # Count how many returned chunks contain the ground truth word
        correct = 0
        for chunk in top_chunks:
            if truth.lower() in chunk.lower():
                correct = correct + 1

        # Precision = correct chunks / k
        precision = correct / k

        # Recall = 1 if at least one correct chunk was found, else 0
        if correct > 0:
            recall = 1
        else:
            recall = 0

        total_precision = total_precision + precision
        total_recall = total_recall + recall

    # Average across all 20 queries
    avg_precision = total_precision / len(test_set)
    avg_recall = total_recall / len(test_set)
    return avg_precision, avg_recall


# ---------- Experiment: try each chunk size with each K ----------
# This tests 3 chunk sizes x 2 K values = 6 combinations,
# so we can see which settings give the best metrics.
print("=== Experiment: chunk sizes and K values ===")
print()

for chunk_size in [300, 500, 700]:
    # Build the database for this chunk size
    collection, num_chunks = build_collection(chunk_size)
    print("Chunk size:", chunk_size, "->", num_chunks, "chunks")

    # Test this chunk size at K=3 and K=5
    for k in [3, 5]:
        avg_precision, avg_recall = evaluate_at_k(collection, k)
        print("   K =", k, "| Precision:", round(avg_precision, 3), "| Recall:", round(avg_recall, 3))
    print()