# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
A complete, hallucination-resistant Retrieval-Augmented Generation (RAG) system with topic modeling, NLP metadata, and a FastAPI web service. It ingests Wikipedia articles, cleans, chunks, embeds, stores them in a vector database, retrieves with semantic search, evaluates retrieval, generates grounded answers with an LLM, checks answers against sources, discovers topics, extracts named entities, and exposes everything through a REST API.

## Week 1 — Environment & Data Acquisition
- Library verification, 5,000-article Wikipedia pipeline, data validation

## Week 2 — Data Cleaning & Preprocessing
- Text cleaning, edge cases, spaCy tokenization/lemmatization, unit tests

## Week 3 — Chunking & Embeddings
- 500-char chunking, unit tests, embeddings (all-MiniLM-L6-v2), timing logs

## Week 4 — Vector Database (ChromaDB)
- ChromaDB setup, ingestion, semantic search, latency tests, edge cases

## Week 5 — Retrieval Evaluation & Optimization
- 20-query test set, Precision@K / Recall@K, chunk-size and K experiments (best: 700, K=5)

## Week 6 — LLM Integration & Prompt Engineering
- OpenRouter API, prompt engineering, error handling, latency logging, CLI

## Week 7 — Hallucination Mitigation & Structured Output
- Hallucination check, stronger prompt, off-topic refusal, JSON output with citations

## Week 8 — NLP Analysis: Topic Modeling
- LDA topic modeling, visualization, manual validation, edge cases, topic-filtered retrieval

## Week 9 — NLP Analysis: Named Entity Recognition
- spaCy NER, evaluation on 50 samples, entity metadata, entity-based retrieval boosting

## Week 10 — API Development (FastAPI)
- The complete RAG system wrapped in a FastAPI application
- Endpoints: `/query` (ask a question), `/metadata` (corpus info), `/health` (health check)
- Request logging (query, chunks, answer, latency) to `api_log.txt`
- Proper HTTP error responses: 400 (empty query), 422 (invalid body, automatic), 500 (internal errors)
- Concurrent-request test confirming the API stays stable under load (5/5 requests succeeded)

## API Details (Week 10)
The RAG system is served with FastAPI and run using uvicorn. It builds the vector database once at startup, then reuses it for every request.

**Endpoints:**
- `GET /health` — returns status and number of chunks in the database
- `GET /metadata` — returns corpus info (articles used, total chunks, chunk size, K, embedding model)
- `POST /query` — takes `{"question": "..."}` and returns the answer, hallucination check, support score, source previews, and latency

**Logging:** Every query is logged to `api_log.txt` with a timestamp, the question, number of chunks, latency, and an answer preview.

**Error handling:** 400 is returned for an empty question, 422 is returned automatically by FastAPI for a malformed request body, and 500 is returned if an internal error (e.g. LLM failure) occurs.

**Stability:** A concurrent test (`test_api.py`) sends 5 requests at the same time using threads. All 5 returned status 200, confirming the API handles simultaneous requests without crashing.

## Evaluation Results (Week 5)
| Chunk Size | K | Precision | Recall |
|------------|---|-----------|--------|
| 300 | 3 | 0.717 | 0.85 |
| 300 | 5 | 0.65 | 0.85 |
| 500 | 3 | 0.70 | 0.80 |
| 500 | 5 | 0.65 | 0.85 |
| 700 | 3 | 0.70 | 0.90 |
| 700 | 5 | 0.70 | 0.95 |

Best configuration: chunk size 700 with K = 5.

## Model Choices
- **Embedding model:** all-MiniLM-L6-v2 (384-dim)
- **LLM:** OpenRouter API
- **Topic model:** LDA (scikit-learn)
- **NER:** spaCy en_core_web_sm
- **API framework:** FastAPI (served with uvicorn)

## Key Files
| File | Description |
|------|-------------|
| `data.py`, `check.py` | Data collection and validation |
| `clean.py`, `test_clean.py`, `nlp_analysis.py` | Cleaning and NLP preprocessing |
| `chunk.py`, `test_chunk.py`, `embed.py`, `test_embed.py` | Chunking and embeddings |
| `vector_db.py`, `test_search.py`, `edge_cases.py` | Vector database and search |
| `evaluate.py` | Precision@K / Recall@K evaluation |
| `rag.py` | RAG with LLM, CLI (Week 6) |
| `rag_v2.py` | Hallucination-resistant RAG with JSON output (Week 7) |
| `topic_model.py`, `topic_filter.py` | Topic modeling and topic-filtered retrieval (Week 8) |
| `ner_extract.py`, `ner_rag.py` | NER and entity-based boosting (Week 9) |
| `api.py` | FastAPI web service (Week 10) |
| `test_api.py` | Concurrent-request stability test (Week 10) |

## Dependencies
- Python 3.13
- wikipedia-api, pandas, spacy, nltk, sentence-transformers, chromadb, openai, scikit-learn, matplotlib, fastapi, uvicorn, requests

## Setup: API Key
The system uses the OpenRouter API. Get a free key at openrouter.ai, create a key, and paste it into the `api_key` field in `api.py` (and `rag.py` / `rag_v2.py`).

## How to Run the API
1. Activate the virtual environment:

venv\Scripts\activate

2. Start the API server:

uvicorn api:app --reload

3. Open the interactive docs in a browser:

http://127.0.0.1:8000/docs

4. In a second terminal, run the concurrent test:

python test_api.py


## Notes
The API builds the database on startup (first 50 articles for speed). The LLM call runs over the network, so latency depends on the API and model speed.