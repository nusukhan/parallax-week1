# RAG Knowledge Extraction System
### Parallax Labs — AI/ML Engineer Internship (12-Week Capstone)

A complete, hallucination-resistant **Retrieval-Augmented Generation (RAG)** system that answers questions from a corpus of Wikipedia science articles. It covers the full pipeline — data collection, cleaning, chunking, embeddings, vector search, retrieval evaluation, LLM generation, hallucination checking, NLP analysis, and a production-style REST API with tests.

---

## Project Overview

**The problem:** Large language models answer from memory, which means they can confidently state things that are wrong. This project solves that by grounding every answer in a real document corpus and verifying the answer against its sources.

**What the system does:**
1. Takes a user question
2. Retrieves the most semantically relevant chunks from a vector database
3. Sends those chunks to an LLM with strict instructions to answer **only** from them
4. Checks the answer against its sources and reports a grounding score
5. Returns a structured JSON response with the answer, sources, and latency

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                          │
└─────────────────────────────────────────────────────────────┘

  Wikipedia API              data.py          5,000 articles
        │                       │                    │
        ▼                       ▼                    ▼
  ┌──────────┐          ┌──────────────┐     ┌──────────────┐
  │ Collect  │  ──────▶ │    Clean     │────▶│    Chunk     │
  │ articles │          │  (clean.py)  │     │  (700 chars) │
  └──────────┘          └──────────────┘     └──────────────┘
                          HTML removal,             │
                          spaCy tokenize            ▼
                                            ┌──────────────┐
                                            │  Embeddings  │
                                            │ MiniLM-L6-v2 │
                                            │  (384-dim)   │
                                            └──────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                          │
│                                                             │
│   ┌────────────────────────────────────────────────┐        │
│   │            ChromaDB Vector Store               │        │
│   │  chunks + embeddings + metadata (topic, NER)   │        │
│   └────────────────────────────────────────────────┘        │
│        │                                                    │
│        │  semantic search (top-K = 5)                       │
│        │  + optional topic filter / entity boost            │
│        ▼                                                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER                          │
│                                                             │
│   retrieved chunks ──▶ prompt (system + context + question) │
│                            │                                │
│                            ▼                                │
│                    LLM (OpenRouter API)                     │
│                            │                                │
│                            ▼                                │
│                  Hallucination check                        │
│           (answer vs sources → support score)               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                     API LAYER (FastAPI)                     │
│                                                             │
│   POST /query     → answer + sources + score + latency      │
│   GET  /metadata  → corpus info                             │
│   GET  /health    → service status                          │
│                                                             │
│   + request logging (api_log.txt)                           │
│   + HTTP errors (400 / 422 / 500)                           │
│   + pytest unit tests + concurrency test                    │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

**Requirements:** Python 3.13, a free [OpenRouter](https://openrouter.ai) API key.

**1. Clone the repository**

git clone https://github.com/nusukhan/parallax-week1.git
cd parallax-week1


**2. Create and activate a virtual environment**

python -m venv venv
venv\Scripts\activate

(On macOS/Linux: `source venv/bin/activate`)

**3. Install dependencies**

pip install -r requirements.txt


**4. Download the spaCy language model**

python -m spacy download en_core_web_sm


**5. Add your API key**

Get a free key at [openrouter.ai](https://openrouter.ai) → Keys → Create Key.
Paste it into the `api_key` field in `api.py`, `rag_v2.py`, and `evaluate_system.py`:
```python
api_key="sk-or-v1-your-key-here"
```

**6. Prepare the data**

The cleaned dataset (`cleaned_articles.csv`) is not included in the repository due to size. Regenerate it with:

python data.py # collects Wikipedia articles (takes 1-2 hours)
python clean.py # produces cleaned_articles.csv


**7. Run the system**

uvicorn api:app --reload

Then open **http://127.0.0.1:8000/docs** in a browser for the interactive API.

---

## Example Queries and Expected Outputs

### Example 1 — A question the corpus can answer

**Request:**
```json
POST /query
{ "question": "what is physics" }
```

**Response:**
```json
{
  "question": "what is physics",
  "answer": "Physics is a natural science that involves the study of matter and its motion through spacetime, along with related concepts such as energy and force. More broadly, it is the general analysis of nature, conducted in order to understand how the universe behaves.",
  "hallucination_check": "Supported",
  "support_score": 0.84,
  "sources": [
    "The following outline is provided as an overview of and topical guide to physics...",
    "Physics Greek physis meaning nature is the natural science which examines..."
  ],
  "latency_seconds": 8.95
}
```

### Example 2 — An off-topic question (correctly refused)

**Request:**
```json
POST /query
{ "question": "how to make biryani" }
```

**Response:**
```json
{
  "question": "how to make biryani",
  "answer": "I don't know based on the provided documents.",
  "hallucination_check": "Refused (correctly said it doesn't know)",
  "support_score": 1.0,
  "latency_seconds": 6.5
}
```

### Example 3 — An empty query (validation error)

**Request:**
```json
POST /query
{ "question": "" }
```

**Response:** `400 Bad Request`
```json
{ "detail": "Question cannot be empty." }
```

### Example 4 — Health check

**Request:** `GET /health`

**Response:**
```json
{ "status": "ok", "chunks_in_db": 440 }
```

---

## Final Performance Benchmarks

### Retrieval evaluation (20 queries, Week 5)

| Chunk Size | K | Precision@K | Recall@K |
|------------|---|-------------|----------|
| 300 | 3 | 0.717 | 0.85 |
| 300 | 5 | 0.65 | 0.85 |
| 500 | 3 | 0.70 | 0.80 |
| 500 | 5 | 0.65 | 0.85 |
| 700 | 3 | 0.70 | 0.90 |
| **700** | **5** | **0.70** | **0.95** |

**Selected configuration:** chunk size 700, K = 5 (highest recall).

### End-to-end evaluation (30 Q&A pairs, Week 11)

| Metric | Result |
|--------|--------|
| Retrieval accuracy | **29/30 (97%)** |
| Generation quality (keyword match) | 21/30 (70%) |
| Average grounding score | 0.41 |
| Correct refusals | 5 |
| Average end-to-end latency | 7.23 s |
| Retrieval-only latency | ~0.002 s |

### API stability

| Test | Result |
|------|--------|
| Concurrent requests (5 simultaneous) | 5/5 succeeded (200 OK) |
| pytest endpoint tests | 6/6 passed |

---

## Weekly Progress

| Week | Focus | Delivered |
|------|-------|-----------|
| 1 | Environment & data | 5,000 Wikipedia articles, validation report |
| 2 | Cleaning | Cleaning functions, edge cases, spaCy, unit tests |
| 3 | Chunking & embeddings | 700-char chunks, MiniLM embeddings, timing logs |
| 4 | Vector database | ChromaDB, semantic search, latency, edge cases |
| 5 | Evaluation | Precision@K / Recall@K, hyperparameter tuning |
| 6 | LLM integration | OpenRouter, prompt engineering, error handling, CLI |
| 7 | Hallucination mitigation | Grounding checks, refusals, JSON + citations |
| 8 | Topic modeling | LDA, visualization, validation, topic filtering |
| 9 | NER | spaCy entities, metadata, retrieval boosting |
| 10 | API | FastAPI endpoints, logging, HTTP errors, concurrency |
| 11 | Testing | 30-pair evaluation, pytest tests, limitations |
| 12 | Documentation | This README, requirements.txt, setup guide |

---

## Known Limitations

- **Corpus subset:** only the first 50 articles are indexed at runtime for speed; questions outside this subset cannot be answered.
- **Grounding metric:** the hallucination check uses word overlap, not semantic similarity, so a correct paraphrase scores low and copied text scores high.
- **No persistence:** the vector store is rebuilt in memory on every startup — this does not scale to production.
- **Fixed-size chunking:** 700-character chunks can split a sentence mid-idea; semantic chunking would preserve meaning better.
- **Free LLM tier:** rate-limited and occasionally returns low-quality or off-format responses.
- **Latency:** dominated by the LLM network call; retrieval itself is milliseconds.
- **NER accuracy:** the general-purpose spaCy model mislabels some technical terms (e.g. "Earthsystem" as PERSON).

---

## Project Structure

| File | Purpose |
|------|---------|
| `data.py`, `check.py` | Collect and validate Wikipedia articles |
| `clean.py`, `test_clean.py`, `nlp_analysis.py` | Cleaning, edge cases, spaCy preprocessing |
| `chunk.py`, `test_chunk.py` | Text chunking + unit tests |
| `embed.py`, `test_embed.py` | Embedding generation + unit test |
| `vector_db.py`, `test_search.py`, `edge_cases.py` | ChromaDB, semantic search, latency, edge cases |
| `evaluate.py` | Precision@K / Recall@K retrieval evaluation |
| `rag.py` | RAG with LLM and CLI (Week 6) |
| `rag_v2.py` | Hallucination-resistant RAG, JSON output, citations |
| `topic_model.py`, `topic_filter.py` | LDA topic modeling and topic-filtered retrieval |
| `ner_extract.py`, `ner_rag.py` | NER extraction and entity-based boosting |
| `api.py` | FastAPI web service |
| `test_api.py` | Concurrent-request stability test |
| `evaluate_system.py` | End-to-end evaluation (30 Q&A pairs) |
| `test_api_endpoints.py` | pytest unit tests for API endpoints |
| `requirements.txt` | All Python dependencies |
| `evaluation_report.txt` | Generated evaluation report |
| `topic_distribution.png` | Topic cluster visualization |
| `entity_metadata.json` | Extracted named entities per article |

---

## Tech Stack

**Language:** Python 3.13
**NLP:** spaCy, NLTK, sentence-transformers (all-MiniLM-L6-v2)
**Vector DB:** ChromaDB
**ML:** scikit-learn (LDA topic modeling)
**LLM:** OpenRouter API
**API:** FastAPI + uvicorn
**Testing:** pytest, FastAPI TestClient
**Data:** pandas, matplotlib

---

*Built by Nusrat Anwar during the Parallax Labs AI/ML Engineer Internship.*