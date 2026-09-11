# ============================================================
# topic_filter.py
#
# WHAT THIS FILE DOES:
# LINE 5: Integrates topic metadata into ChromaDB, so we can
# filter retrieval by topic (search inside only one topic).
#
# Steps:
#   1. Run LDA to find each article's topic
#   2. Chunk the articles, keeping each chunk's topic
#   3. Store chunks in ChromaDB WITH topic as metadata (in batches)
#   4. Search normally, and search filtered by a specific topic
# ============================================================

# pandas reads the cleaned CSV data
import pandas
# chromadb is our vector database
import chromadb
# CountVectorizer turns text into word counts (for LDA)
from sklearn.feature_extraction.text import CountVectorizer
# LatentDirichletAllocation is the LDA topic model
from sklearn.decomposition import LatentDirichletAllocation
# SentenceTransformer turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer

# ---------- Settings ----------
CHUNK_SIZE = 700   # each chunk is 700 characters
NUM_TOPICS = 5     # number of topics for LDA

# ---------- Chunking function ----------
# Splits a long text into pieces of CHUNK_SIZE characters.
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ---------- Load the cleaned data ----------
print("Loading data...")
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(500)
documents = subset["content"].astype(str).tolist()
print("Loaded", len(documents), "documents")

# ============================================================
# STEP 1: Find each article's topic using LDA
# ============================================================
print("Running LDA to find topics...")

# Turn the text into word counts (ignore very common/rare words + stopwords)
vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words="english")
word_counts = vectorizer.fit_transform(documents)

# Run LDA to discover NUM_TOPICS topics
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
lda.fit(word_counts)

# For each article, get a score for every topic, then pick the top one
doc_topic_scores = lda.transform(word_counts)
article_topics = []
for scores in doc_topic_scores:
    article_topics.append(int(scores.argmax()))

print("Each article now has a topic assigned.")

# ============================================================
# STEP 2: Chunk articles, keeping each chunk's topic
# ============================================================
print("Chunking articles and keeping their topics...")

all_chunks = []      # every chunk of text
chunk_topics = []    # the topic number for each chunk

for i in range(len(documents)):
    article = documents[i]
    article_topic = article_topics[i]   # this article's topic

    chunks = chunk_text(article)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_topics.append(article_topic)   # chunk inherits its article's topic

print("Total chunks:", len(all_chunks))

# ============================================================
# STEP 3: Store chunks in ChromaDB WITH topic metadata (in batches)
# ============================================================
print("Creating embeddings and storing in ChromaDB with topic metadata...")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("topic_articles")

# Turn all chunks into embeddings
embeddings = model.encode(all_chunks)
embeddings_list = embeddings.tolist()

# Build a unique ID and metadata (topic) for each chunk
ids = []
metadatas = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))
    metadatas.append({"topic": chunk_topics[i]})   # metadata holds the topic

# ChromaDB has a maximum batch size, so we add chunks 5000 at a time
batch_size = 5000
start = 0
while start < len(all_chunks):
    end = start + batch_size
    collection.add(
        documents=all_chunks[start:end],
        embeddings=embeddings_list[start:end],
        ids=ids[start:end],
        metadatas=metadatas[start:end]
    )
    print("Added chunks", start, "to", min(end, len(all_chunks)))
    start = start + batch_size

print("Stored", collection.count(), "chunks with topic metadata")
print()

# ============================================================
# STEP 4: Search - normal vs topic-filtered
# ============================================================

# The question we want to search for
query = "what is energy"
query_embedding = model.encode([query])

# ---------- Normal search (searches ALL topics) ----------
print("=" * 50)
print("NORMAL SEARCH (all topics) for:", query)
print("=" * 50)

results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
for i in range(len(results["documents"][0])):
    chunk = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    print("[Topic", topic, "]", chunk[:80])

print()

# ---------- Filtered search (searches ONLY one topic) ----------
filter_topic = 2   # only search inside topic 2
print("=" * 50)
print("FILTERED SEARCH (only topic", filter_topic, ") for:", query)
print("=" * 50)

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
    where={"topic": filter_topic}   # THIS line is the topic filter
)
for i in range(len(results["documents"][0])):
    chunk = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    print("[Topic", topic, "]", chunk[:80])