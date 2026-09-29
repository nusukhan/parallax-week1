# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
A complete, hallucination-resistant Retrieval-Augmented Generation (RAG) system with topic modeling, NLP metadata, a FastAPI web service, and full end-to-end evaluation. It ingests Wikipedia articles, cleans, chunks, embeds, stores them in a vector database, retrieves with semantic search, generates grounded answers with an LLM, checks answers against sources, and is tested end to end.

## Weeks 1–10 (Summary)
- **Week 1–2:** Data acquisition (5,000 Wikipedia articles), cleaning, spaCy preprocessing, unit tests
- **Week 3–4:** Chunking, embeddings (all-MiniLM-L6-v2), ChromaDB vector database, semantic search, latency tests
- **Week 5:** Retrieval evaluation (Precision@K / Recall@K), hyperparameter tuning (best: chunk 700, K=5, Recall 0.95)
- **Week 6:** LLM integration (OpenRouter), prompt engineering, API error handling, CLI
- **Week 7:** Hallucination mitigation, off-topic refusal, structured JSON output with citations
- **Week 8:** LDA topic modeling, visualization, validation, topic-filtered retrieval
- **Week 9:** spaCy NER, entity metadata, entity-based retrieval boosting
- **Week 10:** FastAPI service (/query, /metadata, /health), request logging, HTTP errors, concurrency test

## Week 11 — System Evaluation & Testing
- Automated end-to-end evaluation script running a suite of **30 Q&A pairs** (`evaluate_system.py`)
- Comprehensive score report covering retrieval accuracy, generation quality, and latency (`evaluation_report.txt`)
- Known architectural limitations identified and documented
- Unit tests for all FastAPI endpoints using **pytest** and **FastAPI TestClient** (`test_api_endpoints.py`) — 6/6 passing
- Final code cleanup: **docstrings** and **type hints** added throughout

## Evaluation Results (Week 11, 30 Q&A pairs)

| Metric | Result |
|--------|--------|
| Retrieval accuracy (relevant chunk retrieved) | **29/30 (0.97)** |
| Generation quality (answer contains expected keyword) | 21/30 (0.70) |
| Average grounding score (answer vs sources) | 0.41 |
| Correct refusals ("I don't know") | 5 |
| Average end-to-end latency per query | 7.23 seconds |

**Interpretation:** Retrieval is strong (97%), confirming the Week 5 hyperparameter tuning was effective. Generation quality is lower mainly because the LLM paraphrases rather than copying source wording — which lowers the word-overlap grounding score even when the answer is correct. Five refusals were correct behaviour for topics outside the indexed subset.

## Known Limitations
- Only the first 50 articles are indexed, so questions outside this subset cannot be answered.
- The grounding check is word-overlap based, not semantic, so a correct paraphrase can score low.
- The vector database is rebuilt in memory on every startup; there is no persistent store, which does not scale.
- Chunking is fixed-size (700 characters) and can split sentences mid-idea; semantic chunking would preserve meaning better.
- The free LLM tier is rate-limited and occasionally returns low-quality or off-format responses.
- Generation dominates latency (a network call); retrieval itself takes milliseconds.
- General-purpose spaCy NER is imperfect on technical text (e.g. "Earthsystem" tagged as PERSON).

## Testing
Unit tests cover all API endpoints:

| Test | What it checks |
|------|----------------|
| `test_health_returns_ok` | /health returns 200 and a populated database |
| `test_metadata_returns_corpus_info` | /metadata returns the expected fields |
| `test_query_empty_question_returns_400` | Empty question returns 400 |
| `test_query_missing_field_returns_422` | Missing field returns 422 |
| `test_query_wrong_type_returns_422` | Wrong type returns 422 |
| `test_query_valid_question_returns_answer` | Valid query returns answer, sources, latency |

All 6 tests pass.

## Model Choices
- **Embedding model:** all-MiniLM-L6-v2 (384-dim)
- **LLM:** OpenRouter API
- **Topic model:** LDA (scikit-learn)
- **NER:** spaCy en_core_web_sm
- **API framework:** FastAPI (uvicorn)
- **Testing:** pytest + FastAPI TestClient

## Key Files
| File | Description |
|------|-------------|
| `data.py`, `check.py`, `clean.py`, `nlp_analysis.py` | Data collection, validation, cleaning |
| `chunk.py`, `embed.py` | Chunking and embeddings (+ their unit tests) |
| `vector_db.py`, `test_search.py`, `edge_cases.py` | Vector database and search |
| `evaluate.py` | Retrieval evaluation (Precision@K / Recall@K) |
| `rag.py`, `rag_v2.py` | RAG with LLM; hallucination-resistant version with JSON output |
| `topic_model.py`, `topic_filter.py` | Topic modeling and topic-filtered retrieval |
| `ner_extract.py`, `ner_rag.py` | NER and entity-based boosting |
| `api.py` | FastAPI web service |
| `test_api.py` | Concurrent-request stability test |
| `evaluate_system.py` | End-to-end evaluation on 30 Q&A pairs (Week 11) |
| `test_api_endpoints.py` | pytest unit tests for the API endpoints (Week 11) |
| `evaluation_report.txt` | Generated evaluation report |

## Dependencies
Python 3.13 — wikipedia-api, pandas, spacy, nltk, sentence-transformers, chromadb, openai, scikit-learn, matplotlib, fastapi, uvicorn, requests, pytest

## Setup: API Key
Get a free OpenRouter API key at openrouter.ai and paste it into the `api_key` field in `api.py`, `rag.py`, `rag_v2.py`, and `evaluate_system.py`.

## How to Run
1. Activate the virtual environment:

venv\Scripts\activate

2. Run the end-to-end evaluation:

python evaluate_system.py

3. Run the API unit tests:

pytest test_api_endpoints.py -v

4. Start the API server:

uvicorn api:app --reload

Then open `http://127.0.0.1:8000/docs`.

## Notes
The API builds the database on startup (first 50 articles for speed). LLM latency depends on the network and model