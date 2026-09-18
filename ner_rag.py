# ============================================================
# ner_rag.py
#
# LINE 4: Boost search results based on entities.
# We store chunks in ChromaDB WITH their entities as metadata,
# then boost (rank higher) chunks that contain a searched entity.
# ============================================================

import pandas
import chromadb
import spacy
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 700
K = 5

# ---------- Load models ----------
print("Loading models...")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Chunking function ----------
def chunk_text(text, size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

# ---------- Load data ----------
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(100)   # 100 articles (kept small for speed)
print("Loaded", len(subset), "articles")

# ---------- Build chunks + find each chunk's entities ----------
print("Chunking and extracting entities per chunk...")
all_chunks = []
chunk_entity_strings = []   # entities for each chunk (as text)

for content in subset["content"]:
    chunks = chunk_text(str(content))
    for chunk in chunks:
        all_chunks.append(chunk)

        # Run NER on this chunk (first 1000 chars for speed)
        doc = nlp(chunk[:1000])
        words = []
        for entity in doc.ents:
            words.append(entity.text.lower())
        chunk_entity_strings.append(" ".join(words))

print("Total chunks:", len(all_chunks))

# ---------- Store in ChromaDB with entity metadata ----------
print("Creating embeddings and storing with entity metadata...")
client = chromadb.Client()
collection = client.create_collection("ner_articles")

embeddings = model.encode(all_chunks)
embeddings_list = embeddings.tolist()

ids = []
metadatas = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))
    metadatas.append({"entities": chunk_entity_strings[i]})

# Add in batches (ChromaDB batch size limit)
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
    start = start + batch_size

print("Stored", collection.count(), "chunks with entity metadata")
print()

# ============================================================
# LINE 4: Search, then BOOST chunks that contain the entity
# ============================================================
query = "what did Einstein discover"
boost_entity = "einstein"   # the entity we want to boost

query_embedding = model.encode([query])

# Get more results than needed (so we can re-rank them)
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=10)

# Build a list of (chunk, entities) and give a boost if it has the entity
ranked = []
for i in range(len(results["documents"][0])):
    chunk = results["documents"][0][i]
    entities = results["metadatas"][0][i]["entities"]

    # Base score = position (earlier = better). Lower is better.
    score = i

    # BOOST: if the chunk mentions our entity, make its score better (subtract)
    if boost_entity in entities:
        score = score - 100   # big boost = moves to the top

    ranked.append((score, chunk, entities))

# Sort by score (lowest first = best first)
ranked.sort()

print("=" * 50)
print("BOOSTED SEARCH for:", query, "(boosting entity:", boost_entity, ")")
print("=" * 50)
for score, chunk, entities in ranked[:5]:
    boosted = "  <-- BOOSTED" if boost_entity in entities else ""
    print("-", chunk[:80], boosted)