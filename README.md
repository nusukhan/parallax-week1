# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, stores them in a vector database, retrieves relevant content for a user query, and generates grounded answers using an LLM.

## Week 1 — Environment & Data Acquisition

### What I Built
- A verification script confirming all required libraries are installed and working
- An automated data collection pipeline that gathers 5,000 Wikipedia articles
- A data validation script that produces a quality report
- A structured dataset saved as CSV, ready for the next stages of the pipeline

### Approach
Manually listing 5,000 article names is not practical, so I used Wikipedia's category structure instead. Starting from 8 broad science and technology categories, the script extracts sub-categories, collects article names from each, filters out non-article pages (Portal:, Template:, Category:), and downloads the full text of each article.

The download loop is wrapped in error handling so a single failed request does not stop the run, and the dataset is written to disk every 100 articles so progress is never lost if the connection drops.

## Files
| File | Description |
|------|-------------|
| verify.py | Verifies that spaCy, NLTK, pandas, sentence-transformers, and ChromaDB are installed and functional |
| data.py | Collects article names from Wikipedia categories and downloads article content |
| check.py | Validates the dataset and generates the data quality report |
| articles.csv | The resulting dataset (title + full text for each article) |

## Dataset Summary
- Top-level categories: 8
- Sub-categories discovered: 218
- Article names collected: 11,228
- Articles downloaded: 5,000
- Columns: title, content

## Data Quality Report

| Metric | Value |
|--------|-------|
| Total records | 5,000 |
| Columns | title, content |
| Missing fields | 0.00% |
| Duplicate titles | 294 (5.88%) |
| Empty content rows | 0 |
| Average content length | 8,378 characters |

### Findings
- No missing values in either column — every record has both a title and content
- No empty content rows, so all 5,000 articles contain usable text
- 294 duplicate titles (5.88%), caused by articles appearing in more than one Wikipedia category. These are retained in the raw dataset and will be deduplicated during the cleaning stage
- Average article length of ~8,378 characters indicates substantial content suitable for chunking and embedding

## Dependencies
- Python 3.13
- wikipedia-api
- pandas
- spacy
- nltk
- sentence-transformers
- chromadb

## How to Run

1. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
2. Install dependencies:
pip install wikipedia-api pandas spacy nltk sentence-transformers chromadb
python -m spacy download en_core_web_sm
3. Verify the environment:
python verify.py
4. Collect the dataset:
python data.py
5. Validate the dataset:
python check.py
## Notes
Data collection takes approximately 1–2 hours, as each article is fetched individually from the Wikipedia API. A small number of articles are skipped due to transient connection errors; this is expected and handled gracefully.
