# ============================================================
# edge_cases.py
# Tests how ChromaDB behaves in unusual situations (edge cases)
# so that the system does not crash. Two cases are tested:
#   1. Searching an empty database
#   2. Searching with an empty (malformed) query
# ============================================================

# Import ChromaDB (our vector database)
import chromadb

# Import the embedding model
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB client
client = chromadb.Client()

print("=== Testing ChromaDB Edge Cases ===")
print()

# ============================================================
# Edge Case 1: Querying an empty database
# What happens if we search a collection with no chunks in it?
# ============================================================
print("Edge Case 1: Searching an empty database")

# Create a collection but do NOT add anything (it stays empty)
empty_collection = client.create_collection("empty_test")

# try/except protects us: if anything goes wrong, we handle it
try:
    # Turn the question into numbers
    query_embedding = model.encode(["what is physics"])

    # Search the empty database
    results = empty_collection.query(query_embeddings=query_embedding.tolist(), n_results=3)

    # Check if nothing was found
    if len(results["documents"][0]) == 0:
        print("Result: Database is empty, no chunks found (handled safely)")
    else:
        print("Result: Found chunks")
except Exception as e:
    # If an error happens, we catch it instead of crashing
    print("Result: Error handled ->", str(e))

print()

# ============================================================
# Edge Case 2: Empty query (a malformed query)
# What happens if the user searches with an empty string?
# ============================================================
print("Edge Case 2: Searching with an empty query")

# Create a collection and add one chunk to it
test_collection = client.create_collection("test2")
test_collection.add(
    documents=["Physics is the study of matter and energy"],
    embeddings=model.encode(["Physics is the study of matter and energy"]).tolist(),
    ids=["chunk_0"]
)

# try/except protects us again
try:
    # An empty query (nothing typed)
    empty_query = ""

    # Turn the empty query into numbers
    query_embedding = model.encode([empty_query])

    # Search with the empty query
    results = test_collection.query(query_embeddings=query_embedding.tolist(), n_results=3)

    # Report how many chunks came back
    print("Result: Empty query handled, returned", len(results["documents"][0]), "chunks")
except Exception as e:
    # Catch any error instead of crashing
    print("Result: Error handled ->", str(e))

print()
print("=== All edge cases handled without crashing! ===")