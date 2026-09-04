# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a complete, hallucination-resistant Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, cleans them, chunks them, generates embeddings, stores them in a vector database, retrieves relevant chunks using semantic search, evaluates retrieval quality, generates answers with an LLM, checks those answers against the sources, and returns a structured output with citations.

## Week 1 — Environment & Data Acquisition
- Verification script testing all five required libraries
- Automated pipeline collecting 5,000 Wikipedia articles
- Data validation and quality report (with encoding checks)

## Week 2 — Data Cleaning & Preprocessing
- Text cleaning functions: HTML removal, special characters, whitespace
- Edge-case handling: empty, very short, extremely long, mixed-language text
- spaCy tokenization and lemmatization on a subset
- Unit tests for all cleaning functions
- Cleaned dataset with dropped-data percentage logged

## Week 3 — Chunking & Embeddings
- Text chunking using recursive character splitting (500 characters per chunk)
- Unit tests for the chunking function covering edge cases
- Embeddings generated using sentence-transformers (all-MiniLM-L6-v2)
- Embedding generation time logged per chunk

## Week 4 — Vector Database (ChromaDB)
- ChromaDB set up and configured locally
- Chunks and their embeddings ingested into a ChromaDB collection
- Basic semantic search implemented to retrieve the top-K chunks for a query
- Retrieval latency tested and logged for 10 different queries
- Edge cases handled: querying an empty database and empty/malformed queries

## Week 5 — Retrieval Evaluation & Optimization
- Manual test set of 20 queries created, each with an expected ground-truth keyword
- Retrieval evaluation script calculating Precision@K and Recall@K
- Experiments run across different chunk sizes (300, 500, 700) and K values (3, 5)
- Evaluation results documented and the best configuration identified
- Retrieval logic refined to use the best configuration (chunk size 700, K = 5)

## Week 6 — LLM Integration & Prompt Engineering
- Integrated the OpenRouter API to generate answers from retrieved chunks
- Applied prompt engineering: system prompt, context injection, and clear instructions
- Robust error handling for API calls: rate limits, network timeouts, and malformed/empty responses
- End-to-end latency measured and logged (retrieval + generation)
- Simple command-line (CLI) chat interface to interact with the RAG system

## Week 7 — Hallucination Mitigation & Structured Output
- Hallucination check: the generated answer is compared against the source chunks, and a support score is calculated (the fraction of answer words found in the sources)
- Stronger prompt engineering: the model is explicitly told to use only the context, avoid outside knowledge, and reply "I don't know based on the provided documents" if the answer is missing
- Off-topic (out-of-domain) queries are correctly refused instead of answered
- Structured JSON output including the question, answer, hallucination check, support score, source citations, and latency

## Hallucination Mitigation Strategies & Effectiveness
Several strategies were combined to make the system hallucination-resistant:

1. **Strong prompt engineering:** The system prompt instructs the LLM to answer using only the provided context, to not use outside knowledge, and to explicitly say "I don't know based on the provided documents" when the answer is missing.
2. **Answer-vs-source check:** After generation, the answer is compared word-by-word against the retrieved chunks to produce a support score. A low score flags a possibly unsupported (hallucinated) answer.
3. **Refusal detection:** When the model correctly refuses (says it doesn't know), this is recognized as correct behaviour rather than flagged as a hallucination.
4. **Source citations:** Each answer is returned with previews of the source chunks it was based on, so the answer can be verified.

**Observed effectiveness:**
- For an in-domain question like "what is physics", the answer was well grounded, with a support score around 0.74–0.84 and a "Supported" status.
- For an off-topic question like "how to make biryani", the system correctly replied "I don't know based on the provided documents" and was marked as a correct refusal.
- The stronger prompt noticeably increased the support score compared to the earlier version, meaning answers stayed closer to the source documents.

## Evaluation Results (Week 5)
The retrieval system was evaluated on 20 queries using Precision@K and Recall@K:

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
- **Embedding model:** all-MiniLM-L6-v2 — fast, lightweight, produces 384-dimensional embeddings, effective for semantic search.
- **LLM:** accessed through the OpenRouter API for answer generation.

## Files
| File | Description |
|------|-------------|
| `verify.py` | Verifies all five libraries |
| `data.py` | Collects and downloads Wikipedia articles |
| `check.py` | Validates the dataset and generates the quality report |
| `clean.py` | Cleaning functions and edge-case handling |
| `test_clean.py` | Unit tests for cleaning functions |
| `nlp_analysis.py` | spaCy tokenization and lemmatization |
| `chunk.py` | Text chunking function |
| `test_chunk.py` | Unit tests for the chunking function |
| `embed.py` | Generates embeddings and logs performance |
| `test_embed.py` | Unit test for embedding generation |
| `vector_db.py` | Sets up ChromaDB, ingests chunks, and runs semantic search (best settings: chunk size 700, K=5) |
| `test_search.py` | Tests retrieval latency for 10 queries |
| `edge_cases.py` | Handles ChromaDB edge cases |
| `evaluate.py` | Evaluates retrieval with Precision@K and Recall@K, and experiments with chunk sizes and K values |
| `rag.py` | RAG system with LLM integration, error handling, latency logging, and a CLI (Week 6) |
| `rag_v2.py` | Hallucination-resistant RAG: answer-vs-source check, stronger prompt, off-topic refusal, and structured JSON output with citations (Week 7) |

## Dependencies
- Python 3.13
- wikipedia-api, pandas, spacy, nltk, sentence-transformers, chromadb, openai

## Setup: API Key
The RAG system uses the OpenRouter API. To run `rag.py` or `rag_v2.py`, you need a free OpenRouter API key:
1. Sign up at openrouter.ai
2. Create an API key
3. Paste it into the `api_key` field in the file

## How to Run

1. Activate the virtual environment:
venv\Scripts\activate


2. Verify the environment:

python verify.py


3. Collect the dataset:

python data.py


4. Validate the dataset:

python check.py


5. Clean the dataset:

python clean.py


6. Run the cleaning tests:

python test_clean.py


7. Run the chunking tests:

python test_chunk.py


8. Generate embeddings:

python embed.py


9. Run the embedding test:

python test_embed.py


10. Set up the vector database and run semantic search:

python vector_db.py


11. Test retrieval latency:

python test_search.py


12. Test edge cases:

python edge_cases.py


13. Run the retrieval evaluation:

python evaluate.py


14. Run the Week 6 RAG system (CLI chat):

python rag.py


15. Run the Week 7 hallucination-resistant RAG (CLI chat with JSON output):

python rag_v2.py


## Notes
Data collection takes 1–2 hours as each article is fetched individually from the Wikipedia API. Skipped articles due to connection errors are expected and handled gracefully. The LLM call runs over the network, so generation latency depends on the API and model speed.