# ============================================================
# evaluate_system.py
#
# WHAT THIS FILE DOES:
# Runs an automated END-TO-END evaluation of the full RAG system
# on a suite of 30 Q&A pairs, then prints and saves a score report
# covering three things:
#   1. Retrieval accuracy  — did we retrieve chunks containing the answer?
#   2. Generation quality  — is the generated answer correct and grounded?
#   3. Latency             — how long does each query take end to end?
#
# It also documents the known limitations of the current architecture.
# The full report is written to evaluation_report.txt
# ============================================================

# time -> measure how long each query takes
import time
# typing -> type hints, so it is clear what each function takes and returns
from typing import List, Dict

# pandas -> read the CSV data
import pandas
# chromadb -> our vector database
import chromadb
# SentenceTransformer -> turns text into embeddings (numbers)
from sentence_transformers import SentenceTransformer
# OpenAI -> used to talk to the LLM through OpenRouter
from openai import OpenAI

# ---------- Settings ----------
CHUNK_SIZE: int = 700   # each chunk is 700 characters (best from Week 5)
K: int = 5              # retrieve the top 5 chunks per query


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Split a long text into pieces of `size` characters.

    Args:
        text: The full article text.
        size: How many characters each chunk should hold.

    Returns:
        A list of text chunks.
    """
    chunks: List[str] = []
    # Walk through the text in steps of `size` characters
    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])
    return chunks


# ============================================================
# Build the RAG system once (database + LLM client)
# ============================================================
print("Setting up the RAG system...")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a fresh ChromaDB collection for this evaluation
client_db = chromadb.Client()
collection = client_db.create_collection("eval_articles")

# Read data and build chunks (first 50 articles to stay fast)
df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(50)
all_chunks: List[str] = []
for content in subset["content"]:
    all_chunks = all_chunks + chunk_text(str(content))

# Turn chunks into embeddings and give each a unique ID
embeddings = model.encode(all_chunks)
ids: List[str] = []
for i in range(len(all_chunks)):
    ids.append("chunk_" + str(i))

# Store everything in ChromaDB
collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=ids)
print("Database ready with", collection.count(), "chunks")
print()

# The LLM client (key is kept as a placeholder for security)
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="PASTE_YOUR_KEY_HERE"
)


# ============================================================
# The evaluation suite: 30 Q&A pairs
# Each pair has a question and a keyword that a correct answer
# (and a correctly retrieved chunk) should contain.
# ============================================================
qa_pairs: List[Dict[str, str]] = [
    {"question": "what is physics", "keyword": "physics"},
    {"question": "what is energy", "keyword": "energy"},
    {"question": "what is an atom", "keyword": "atom"},
    {"question": "what is light", "keyword": "light"},
    {"question": "how do stars form", "keyword": "star"},
    {"question": "what is gravity", "keyword": "gravit"},
    {"question": "what is a wave", "keyword": "wave"},
    {"question": "what is matter", "keyword": "matter"},
    {"question": "what is force", "keyword": "force"},
    {"question": "what is motion", "keyword": "motion"},
    {"question": "what is heat", "keyword": "heat"},
    {"question": "what is temperature", "keyword": "temperature"},
    {"question": "what is a particle", "keyword": "particle"},
    {"question": "what is quantum mechanics", "keyword": "quantum"},
    {"question": "what is relativity", "keyword": "relativ"},
    {"question": "who was Albert Einstein", "keyword": "einstein"},
    {"question": "who was Isaac Newton", "keyword": "newton"},
    {"question": "what is a magnetic field", "keyword": "magnetic"},
    {"question": "what is electricity", "keyword": "electric"},
    {"question": "what is a molecule", "keyword": "molecul"},
    {"question": "what is radiation", "keyword": "radiation"},
    {"question": "what is nuclear physics", "keyword": "nuclear"},
    {"question": "what is thermodynamics", "keyword": "thermodynam"},
    {"question": "what is mass", "keyword": "mass"},
    {"question": "what is velocity", "keyword": "velocit"},
    {"question": "what is a vector", "keyword": "vector"},
    {"question": "what is reflection of light", "keyword": "reflect"},
    {"question": "what is refraction", "keyword": "refract"},
    {"question": "what is a spectrum", "keyword": "spectr"},
    {"question": "what is classical mechanics", "keyword": "mechanic"},
]


def retrieve(question: str) -> List[str]:
    """Retrieve the top-K most relevant chunks for a question.

    Args:
        question: The user's question.

    Returns:
        A list of the top-K chunk texts.
    """
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(), n_results=K
    )
    return results["documents"][0]


def generate(question: str, chunks: List[str]) -> str:
    """Ask the LLM to answer the question using only the given chunks.

    Args:
        question: The user's question.
        chunks: The retrieved context chunks.

    Returns:
        The LLM's answer as text.
    """
    # Join the chunks into one context block
    context: str = ""
    for chunk in chunks:
        context = context + chunk + "\n\n"

    # Strict prompt: only use the context, otherwise say "I don't know"
    system_prompt: str = (
        "You are a helpful assistant. Answer using ONLY the context. "
        "If the answer is not in the context, say: "
        "'I don't know based on the provided documents.'"
    )
    response = llm_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Context:\n" + context + "\nQuestion: " + question}
        ],
        timeout=30
    )
    return response.choices[0].message.content


def check_grounded(answer: str, chunks: List[str]) -> float:
    """Measure how much of the answer is supported by the source chunks.

    Args:
        answer: The generated answer.
        chunks: The chunks the answer was based on.

    Returns:
        The fraction (0.0 to 1.0) of the answer's long words found
        in the sources. Higher means better grounded.
    """
    # Join all chunks into one lowercased block of text
    source_text: str = ""
    for chunk in chunks:
        source_text = source_text + chunk.lower() + " "

    # Only check longer words (skip "the", "is", etc.)
    supported: int = 0
    total: int = 0
    for word in answer.lower().split():
        if len(word) > 4:
            total = total + 1
            if word in source_text:
                supported = supported + 1

    if total == 0:
        return 0.0
    return supported / total


# ============================================================
# Run the evaluation over all 30 Q&A pairs
# ============================================================
print("Running end-to-end evaluation on", len(qa_pairs), "Q&A pairs...")
print()

retrieval_hits: int = 0        # how many queries retrieved a relevant chunk
answer_hits: int = 0           # how many answers contained the keyword
refusals: int = 0              # how many times the system said "I don't know"
total_latency: float = 0.0     # sum of all query times
total_grounding: float = 0.0   # sum of all grounding scores
report_lines: List[str] = []   # per-query lines for the saved report

for pair in qa_pairs:
    question = pair["question"]
    keyword = pair["keyword"]

    # Start timing this query (end-to-end: retrieval + generation)
    start = time.time()
    chunks = retrieve(question)

    # --- RETRIEVAL ACCURACY: did any retrieved chunk contain the keyword? ---
    retrieval_ok = False
    for chunk in chunks:
        if keyword.lower() in chunk.lower():
            retrieval_ok = True
    if retrieval_ok:
        retrieval_hits = retrieval_hits + 1

    # --- GENERATION: ask the LLM (wrapped so one failure doesn't stop the run) ---
    try:
        answer = generate(question, chunks)
    except Exception as e:
        answer = "ERROR: " + str(e)

    # --- LATENCY: total time for this query ---
    latency = time.time() - start
    total_latency = total_latency + latency

    # --- GENERATION QUALITY: grounding score + keyword presence ---
    grounding = check_grounded(answer, chunks)
    total_grounding = total_grounding + grounding

    answer_ok = keyword.lower() in answer.lower()
    if answer_ok:
        answer_hits = answer_hits + 1
    if "i don't know" in answer.lower():
        refusals = refusals + 1

    # Build one readable line for this query
    line = (
        question
        + " | retrieval: " + ("OK" if retrieval_ok else "MISS")
        + " | answer: " + ("OK" if answer_ok else "MISS")
        + " | grounding: " + str(round(grounding, 2))
        + " | latency: " + str(round(latency, 2)) + "s"
    )
    print(line)
    report_lines.append(line)


# ============================================================
# Build the final score report
# ============================================================
n = len(qa_pairs)
retrieval_accuracy = retrieval_hits / n
answer_accuracy = answer_hits / n
avg_grounding = total_grounding / n
avg_latency = total_latency / n

report: List[str] = []
report.append("=" * 60)
report.append("END-TO-END EVALUATION REPORT (" + str(n) + " Q&A pairs)")
report.append("=" * 60)
report.append("")
report.append("1. RETRIEVAL ACCURACY")
report.append("   Queries where a relevant chunk was retrieved: "
              + str(retrieval_hits) + "/" + str(n)
              + " (" + str(round(retrieval_accuracy, 2)) + ")")
report.append("")
report.append("2. GENERATION QUALITY")
report.append("   Answers containing the expected keyword: "
              + str(answer_hits) + "/" + str(n)
              + " (" + str(round(answer_accuracy, 2)) + ")")
report.append("   Average grounding score (answer vs sources): "
              + str(round(avg_grounding, 2)))
report.append("   Correct refusals ('I don't know'): " + str(refusals))
report.append("")
report.append("3. LATENCY")
report.append("   Average end-to-end latency per query: "
              + str(round(avg_latency, 2)) + " seconds")
report.append("   Total time for all queries: "
              + str(round(total_latency, 2)) + " seconds")
report.append("")
report.append("=" * 60)
report.append("KNOWN LIMITATIONS OF THE CURRENT ARCHITECTURE")
report.append("=" * 60)
report.append("- Only the first 50 articles are indexed, so questions about")
report.append("  topics outside this subset cannot be answered.")
report.append("- The grounding check is word-overlap based, not semantic, so a")
report.append("  correct paraphrase can score low and copied text can score high.")
report.append("- The database is rebuilt in memory on every startup; there is no")
report.append("  persistent vector store, which does not scale.")
report.append("- Chunking is fixed-size (700 chars) and can split sentences")
report.append("  mid-idea; semantic chunking would preserve meaning better.")
report.append("- The free LLM tier is rate-limited and occasionally returns")
report.append("  low-quality or off-format responses.")
report.append("- Generation dominates latency (a network call); retrieval itself")
report.append("  takes only milliseconds.")
report.append("- NER labels from the general-purpose spaCy model are imperfect on")
report.append("  technical text (e.g. 'Earthsystem' tagged as PERSON).")

# Print the report
print()
for line in report:
    print(line)

# ---------- Save the report to a file ----------
out = open("evaluation_report.txt", "w", encoding="utf-8")
for line in report_lines:
    out.write(line + "\n")
out.write("\n")
for line in report:
    out.write(line + "\n")
out.close()

print()
print("Report saved to evaluation_report.txt")