# Import tools
from sentence_transformers import SentenceTransformer
from chunk import chunk_text
import pandas
import time

# Load the embedding model (turns each chunk into 384 numbers)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the real cleaned dataset
df = pandas.read_csv("cleaned_articles.csv")

# Take a subset of 50 articles for embedding (to keep it fast)
subset = df.head(50)

# Break all articles into chunks
all_chunks = []
for content in subset["content"]:
    chunks = chunk_text(str(content))
    all_chunks = all_chunks + chunks

print("Total chunks from 50 articles:", len(all_chunks))

# Measure the time taken to generate embeddings
start = time.time()
embeddings = model.encode(all_chunks)
end = time.time()

# Calculate timing
total_time = end - start
time_per_chunk = total_time / len(all_chunks)

print("Size of one embedding:", len(embeddings[0]))
print("Total time:", round(total_time, 2), "seconds")
print("Time per chunk:", round(time_per_chunk, 4), "seconds")

# Estimate time for all chunks in the full dataset
# 4961 articles, roughly the same chunks-per-article ratio
chunks_per_article = len(all_chunks) / 50
total_articles = 4961
expected_chunks = int(chunks_per_article * total_articles)
expected_time = time_per_chunk * expected_chunks
print("Estimated total chunks for full dataset:", expected_chunks)
print("Estimated total embedding time:", round(expected_time, 2), "seconds")