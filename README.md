# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, cleans them, chunks them, generates embeddings, stores them in a vector database, retrieves relevant chunks using semantic search, and evaluates retrieval quality.

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

## Evaluation Results
The retrieval system was evaluated on 20 queries using Precision@K and Recall@K.
Different chunk sizes and K values were tested:

| Chunk Size | K | Precision | Recall |
|------------|---|-----------|--------|
| 300 | 3 | 0.717 | 0.85 |
| 300 | 5 | 0.65 | 0.85 |
| 500 | 3 | 0.70 | 0.80 |
| 500 | 5 | 0.65 | 0.85 |
| 700 | 3 | 0.70 | 0.90 |
| 700 | 5 | 0.70 | 0.95 |

## Impact of Hyperparameter Changes
- **Chunk size:** Larger chunks (700 characters) produced better recall, because each chunk holds more context and is more likely to contain the answer. Smaller chunks (300) created more total chunks but did not improve recall.
- **K value:** A higher K (5) increased recall, since checking more chunks gives more chances to find a correct one.
- **Best configuration:** Chunk size 700 with K = 5 gave the best results (Precision 0.70, Recall 0.95).

## System Optimization
Based on the evaluation findings, the retrieval logic in `vector_db.py` was updated to use the best configuration found: chunk size 700 and K = 5. This improves recall while keeping precision stable.

## Chunking Strategy Decision
I chose recursive character splitting because it is simple, reliable, and easy to test. After evaluation, a chunk size of 700 characters was found to give the best retrieval performance.

## Model Choice
I chose all-MiniLM-L6-v2 because it is fast, lightweight, and runs easily on a normal laptop. It produces 384-dimensional embeddings, which are compact but effective for semantic search.

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
| `vector_db.py` | Sets up ChromaDB, ingests chunks, and runs semantic search (uses best settings: chunk size 700, K=5) |
| `test_search.py` | Tests retrieval latency for 10 queries |
| `edge_cases.py` | Handles ChromaDB edge cases |
| `evaluate.py` | Evaluates retrieval with Precision@K and Recall@K, and experiments with chunk sizes and K values |

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


13. Run the retrieval evaluation:

python evaluate.py


## Notes
Data collection takes 1–2 hours as each article is fetched individually from the Wikipedia API. Skipped articles due to connection errors are expected and handled gracefully.