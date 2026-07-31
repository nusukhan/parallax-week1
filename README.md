# Parallax Labs Internship — RAG Knowledge Extraction System

## Project Overview
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a large collection of real-world documents. The system ingests Wikipedia articles, cleans them, and prepares them for embedding and retrieval.

## Week 1 — Environment & Data Acquisition
- Verification script testing all five required libraries (spaCy, NLTK, pandas, sentence-transformers, ChromaDB)
- Automated pipeline collecting 5,000 Wikipedia articles from science and technology categories
- Data validation and quality report (including encoding checks)

## Week 2 — Data Cleaning & Preprocessing
- Text cleaning functions: remove HTML tags, remove special characters, normalize whitespace
- Edge-case handling: empty text, very short text, extremely long text (truncated), and mixed-language text (non-English articles dropped)
- spaCy used for tokenization and lemmatization on a subset of the data
- Unit tests for all cleaning functions
- Cleaned dataset saved, with the percentage of dropped data logged

## Files
| File | Description |
|------|-------------|
| `verify.py` | Verifies all five libraries are installed and functional |
| `data.py` | Collects article names from Wikipedia categories and downloads content |
| `check.py` | Validates the dataset and generates the data quality report (with encoding check) |
| `clean.py` | Cleaning functions, edge-case handling, and the script that produces the cleaned dataset |
| `test_clean.py` | Unit tests for the cleaning functions |
| `nlp_analysis.py` | spaCy tokenization and lemmatization on a subset of articles |
| `articles.csv` | Raw dataset (title + content) |
| `cleaned_articles.csv` | Cleaned dataset ready for embedding |

## Cleaning Results
- Articles before cleaning: 5,000
- Articles after cleaning: 4,961
- Data dropped: 39 articles (0.78%)

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


6. Run the unit tests:

python test_clean.py


7. Run the spaCy NLP analysis:

python nlp_analysis.py


## Notes
Data collection takes 1–2 hours as each article is fetched individually from the Wikipedia API. A small number of articles are skipped due to transient connection errors; this is expected and handled gracefully.