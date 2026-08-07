from sentence_transformers import SentenceTransformer
from chunk import chunk_text

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test: embeddings are generated correctly
text = "AI is great. " * 100
chunks = chunk_text(text)
embeddings = model.encode(chunks)

# Check: number of embeddings equals number of chunks
assert len(embeddings) == len(chunks)

# Check: each embedding has 384 numbers
assert len(embeddings[0]) == 384

print("All embedding tests passed!")