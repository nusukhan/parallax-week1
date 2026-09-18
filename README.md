# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a complete, hallucination-resistant Retrieval-Augmented Generation (RAG) system with topic modeling and NLP metadata enrichment. It ingests Wikipedia articles, cleans them, chunks them, generates embeddings, stores them in a vector database, retrieves relevant chunks with semantic search, evaluates retrieval, generates grounded answers with an LLM, checks answers against sources, discovers topics, extracts named entities, and supports entity-based retrieval boosting.

## Week 1 — Environment & Data Acquisition
- Verification script testing all five required libraries
- Automated pipeline collecting 5,000 Wikipedia articles
- Data validation and quality report

## Week 2 — Data Cleaning & Preprocessing
- Text cleaning, edge-case handling, spaCy tokenization/lemmatization, unit tests

## Week 3 — Chunking & Embeddings
- 500-character chunking, unit tests, embeddings (all-MiniLM-L6-v2), timing logs

## Week 4 — Vector Database (ChromaDB)
- ChromaDB setup, chunk ingestion, semantic search, latency tests, edge cases

## Week 5 — Retrieval Evaluation & Optimization
- 20-query test set, Precision@K / Recall@K, chunk-size and K experiments (best: 700, K=5)

## Week 6 — LLM Integration & Prompt Engineering
- OpenRouter API, prompt engineering, API error handling, latency logging, CLI

## Week 7 — Hallucination Mitigation & Structured Output
- Hallucination check, stronger prompt, off-topic refusal, JSON output with citations

## Week 8 — NLP Analysis: Topic Modeling
- LDA topic modeling, bar-chart visualization, manual validation, edge-case handling, topic-filtered retrieval

## Week 9 — NLP Analysis: Named Entity Recognition (NER)
- Named Entity Recognition using spaCy to extract people, places, dates, and organizations from the corpus
- Evaluation of NER on 50 samples (detection rate measured)
- Entities extracted per chunk and stored as metadata (entity_metadata.json and in ChromaDB)
- Retrieval boosting: chunks containing a searched entity are ranked higher
- Accuracy and usefulness of the entity metadata documented below

## NER Details (Week 9)
Named Entity Recognition was performed with spaCy's `en_core_web_sm` model. For each article, spaCy identifies entities such as PERSON (e.g. "Isaac Newton", "Albert Einstein"), GPE/LOC (places like "Germany", "Earth"), DATE (e.g. "1905"), and ORG (organizations).

**Accuracy:** On a sample of 50 articles, 50/50 produced usable entities (a 1.0 detection rate). Spot-checking showed most entities were correct, though NER is not perfect — for example, some tokens like "Earthsystem" were mislabeled as PERSON. This is a known limitation of general-purpose NER on technical text.

**Usefulness:** The extracted entities are stored as metadata alongside each chunk in ChromaDB. This enables entity-based retrieval boosting: when a user searches for a specific entity (e.g. "Einstein"), chunks that mention that entity are ranked higher. In testing, a search for "what did Einstein discover" correctly boosted all Einstein-related chunks to the top.

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
- **Embedding model:** all-MiniLM-L6-v2 (384-dimensional embeddings)
- **LLM:** OpenRouter API for answer generation
- **Topic model:** LDA (scikit-learn)
- **NER:** spaCy en_core_web_sm

## Files
| File | Description |
|------|-------------|
| `verify.py` | Verifies all five libraries |
| `data.py` | Collects Wikipedia articles |
| `check.py` | Validates the dataset |
| `clean.py` | Cleaning functions and edge cases |
| `test_clean.py` | Unit tests for cleaning |
| `nlp_analysis.py` | spaCy tokenization and lemmatization |
| `chunk.py` | Text chunking function |
| `test_chunk.py` | Unit tests for chunking |
| `embed.py` | Generates embeddings and logs performance |
| `test_embed.py` | Unit test for embedding generation |
| `vector_db.py` | ChromaDB setup and semantic search |
| `test_search.py` | Retrieval latency tests |
| `edge_cases.py` | ChromaDB edge cases |
| `evaluate.py` | Precision@K and Recall@K evaluation |
| `rag.py` | RAG with LLM, error handling, latency, CLI (Week 6) |
| `rag_v2.py` | Hallucination-resistant RAG with JSON output (Week 7) |
| `topic_model.py` | LDA topic modeling, visualization, validation (Week 8) |
| `topic_filter.py` | Topic-filtered retrieval (Week 8) |
| `ner_extract.py` | NER extraction, evaluation, and metadata saving (Week 9) |
| `ner_rag.py` | Entity metadata in ChromaDB + entity-based retrieval boosting (Week 9) |
| `topic_distribution.png` | Bar chart of documents per topic |
| `topic_validation.txt` | 20 random documents per topic |
| `entity_metadata.json` | Extracted entities per article |

## Dependencies
- Python 3.13
- wikipedia-api, pandas, spacy, nltk, sentence-transformers, chromadb, openai, scikit-learn, matplotlib

## Setup: API Key
To run `rag.py` or `rag_v2.py`, get a free OpenRouter API key at openrouter.ai, create a key, and paste it into the `api_key` field in the file.

## How to Run
1. Activate the virtual environment:
venv\Scripts\activate

2. Run NER extraction and evaluation:

python ner_extract.py

3. Run entity-based retrieval boosting:

python ner_rag.py

4. Run the hallucination-resistant RAG (CLI chat):

python rag_v2.py

Other scripts (Weeks 1–8) run the same way, e.g. `python topic_model.py`, `python vector_db.py`, `python evaluate.py`.

## Notes
NLP tasks run on a subset of the corpus for speed. Chunks are added to ChromaDB in batches due to its max batch size. The LLM call runs over the network, so generation latency depends on the API.