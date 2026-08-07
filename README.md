# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, cleans them, chunks them, and generates embeddings for retrieval.

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
- Unit tests for the chunking function covering edge cases (very short text, text without clear sentence boundaries)
- Embeddings generated using sentence-transformers (all-MiniLM-L6-v2) on real cleaned articles
- Embedding generation time logged per chunk, with total expected indexing time calculated

## Chunking Strategy Decision
I chose recursive character splitting (fixed 500-character chunks) because it is simple, reliable, and works well for a first version. Semantic chunking was considered but not used, as it is more complex and not needed at this stage. Fixed-size chunks are easy to test and give predictable results.

## Model Choice
I chose all-MiniLM-L6-v2 because it is fast, lightweight, and runs easily on a normal laptop. It produces 384-dimensional embeddings, which are compact but effective for semantic search. It is a well-established, widely-used model, making it a safe and reliable choice.

## Performance Logs
Embeddings were generated on a subset of 50 real articles from the cleaned dataset:
- Chunks from 50 articles: 607
- Embedding dimension: 384
- Time per chunk: ~0.029 seconds
- Estimated total chunks for full dataset (4,961 articles): 60,226
- Estimated total embedding time: ~1,746 seconds (approximately 29 minutes)

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
| `embed.py` | Generates embeddings on real data and logs performance |
| `test_embed.py` | Unit test for embedding generation |

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


## Notes
Data collection takes 1–2 hours as each article is fetched individually from the Wikipedia API. Skipped articles due to connection errors are expected and handled gracefully.