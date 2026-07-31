import spacy
import nltk
import pandas
import sentence_transformers
import chromadb

print("=== Testing all libraries ===")

# 1. spaCy test
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world")
print("spaCy works, tokens:", len(doc))

# 2. NLTK test
nltk.download("punkt_tab")
tokens = nltk.word_tokenize("Hello world")
print("NLTK works, tokens:", len(tokens))

# 3. pandas test
test_df = pandas.DataFrame({"col": [1, 2, 3]})
print("pandas works, rows:", len(test_df))

# 4. sentence-transformers test
model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("Hello world")
print("sentence-transformers works, vector size:", len(vector))

# 5. chromadb test
client = chromadb.Client()
collection = client.create_collection("test")
print("chromadb works, collection created")

print("=== All 5 libraries are working! ===")