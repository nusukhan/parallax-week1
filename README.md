# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, cleans them, chunks them, generates embeddings, stores them in a vector database, and retrieves relevant chunks using semantic search.

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

## Chunking Strategy Decision
I chose recursive character splitting (fixed 500-character chunks) because it is simple, reliable, and works well for a first version. Semantic chunking was considered but not used, as it is more complex and not needed at this stage.

## Model Choice
I chose all-MiniLM-L6-v2 because it is fast, lightweight, and runs easily on a normal laptop. It produces 384-dimensional embeddings, which are compact but effective for semantic search.

## Performance Logs
- Chunks from 50 articles: 607
- Embedding dimension: 384
- Average retrieval time per query: ~0.0023 seconds (tested on 10 queries)

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
| `vector_db.py` | Sets up ChromaDB, ingests chunks, and runs semantic search |
| `test_search.py` | Tests retrieval latency for 10 queries |
| `edge_cases.py` | Handles ChromaDB edge cases |

## Dependencies
- Python 3.13
- wikipedia-api, pandas, spacy, nltk, sentence-transformers, chromadb

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


## Notes
Data collection takes 1–2 hours as each article is fetched individually from the Wikipedia API. Skipped articles due to connection errors are expected and handled gracefully