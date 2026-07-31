import pandas
import spacy

nlp = spacy.load("en_core_web_sm")

df = pandas.read_csv("cleaned_articles.csv")
subset = df.head(10)

print("=== spaCy NLP Analysis on 10 articles ===\n")

for index, row in subset.iterrows():
    text = str(row["content"])[:200]
    doc = nlp(text)

    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]

    print("Article:", row["title"])
    print("Tokens (first 10):", tokens[:10])
    print("Lemmas (first 10):", lemmas[:10])
    print("-" * 40)