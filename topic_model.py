# ============================================================
# topic_model.py
#
# WHAT THIS FILE DOES:
# Applies topic modeling (LDA) to the article corpus to discover
# the hidden themes/topics inside the documents.
#
# Parts:
#   LINE 1: Find topics using LDA and show the top words per topic
#   LINE 2: Count documents per topic and draw a bar chart
#   LINE 3: Show 20 random documents per topic for manual validation
#   LINE 4: Handle edge cases (short documents + heavy jargon)
#
# LDA = Latent Dirichlet Allocation, a classic topic modeling method.
# It groups words that often appear together into "topics".
# ============================================================

import pandas
# CountVectorizer turns text into word counts (numbers)
from sklearn.feature_extraction.text import CountVectorizer
# LatentDirichletAllocation is the LDA topic model
from sklearn.decomposition import LatentDirichletAllocation
# matplotlib is used to draw the bar chart
import matplotlib.pyplot as plt
# random is used to pick random documents for validation
import random

# ---------- Load the cleaned data ----------
print("Loading data...")
df = pandas.read_csv("cleaned_articles.csv")

# Use a subset to keep it fast (you can increase this later)
subset = df.head(500)
documents = subset["content"].astype(str).tolist()
print("Loaded", len(documents), "documents")
print()

# ============================================================
# LINE 4: HANDLE EDGE CASES
# 1. Count and flag extremely short documents (very little text).
# 2. Remove heavy jargon (LaTeX/math codes like "mathbf", "displaystyle")
#    so they don't pollute the topics.
# ============================================================

# ---------- Edge case 1: extremely short documents ----------
short_docs = 0
for doc in documents:
    if len(doc) < 100:   # fewer than 100 characters = very short
        short_docs = short_docs + 1
print("Edge case check: found", short_docs, "extremely short documents (< 100 chars)")

# ---------- Edge case 2: remove heavy jargon / math codes ----------
# These LaTeX/math tokens are not real topic words, so we remove them.
jargon_words = ["mathbf", "displaystyle", "frac", "mathrm", "partial", "nabla", "cdot", "sqrt"]

cleaned_documents = []
for doc in documents:
    words = doc.split()
    kept_words = []
    for word in words:
        # Keep the word only if it is NOT a jargon word
        if word.lower() not in jargon_words:
            kept_words.append(word)
    cleaned_documents.append(" ".join(kept_words))

# Use the cleaned documents from now on
documents = cleaned_documents
print("Edge case handling done: removed heavy jargon/math codes")
print()

# ============================================================
# LINE 1: TOPIC MODELING WITH LDA
# ============================================================

# ---------- Turn text into word counts ----------
# max_df=0.95  -> ignore words that appear in 95%+ of docs (too common)
# min_df=5     -> ignore words that appear in fewer than 5 docs (too rare)
# stop_words   -> remove common English words like "the", "is"
print("Converting text to word counts...")
vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words="english")
word_counts = vectorizer.fit_transform(documents)

# ---------- Apply LDA topic modeling ----------
# n_components=5 -> find 5 topics (you can change this number)
print("Finding topics with LDA...")
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(word_counts)

print("Done! Found", num_topics, "topics.")
print()

# ---------- Show the top words for each topic ----------
# Each topic is described by its most important words.
feature_names = vectorizer.get_feature_names_out()

print("=" * 50)
print("DISCOVERED TOPICS (top 10 words each)")
print("=" * 50)

for topic_number in range(num_topics):
    # Get the word importance scores for this topic
    topic = lda.components_[topic_number]

    # Find the top 10 words (highest scores) for this topic
    top_word_indexes = topic.argsort()[-10:]

    # Get the actual words
    top_words = []
    for index in top_word_indexes:
        top_words.append(feature_names[index])

    print()
    print("Topic", topic_number, ":", ", ".join(top_words))

# ============================================================
# LINE 2: VISUALIZE THE TOPIC CLUSTERS
# Count how many documents belong to each topic, then draw a bar chart.
# ============================================================
print()
print("Counting documents per topic...")

# lda.transform gives each document a score for every topic
doc_topic_scores = lda.transform(word_counts)

# Count how many documents have each topic as their top topic
topic_counts = [0] * num_topics
for scores in doc_topic_scores:
    # Find the topic with the highest score for this document
    best_topic = scores.argmax()
    topic_counts[best_topic] = topic_counts[best_topic] + 1

# Print the counts
for topic_number in range(num_topics):
    print("Topic", topic_number, "->", topic_counts[topic_number], "documents")

# ---------- Draw the bar chart ----------
topic_labels = []
for i in range(num_topics):
    topic_labels.append("Topic " + str(i))

plt.figure(figsize=(8, 5))
plt.bar(topic_labels, topic_counts)
plt.xlabel("Topics")
plt.ylabel("Number of documents")
plt.title("Number of Documents per Topic")

# Save the chart as an image file
plt.savefig("topic_distribution.png")
print()
print("Chart saved as topic_distribution.png")

# Show the chart in a window (close it to continue)
plt.show()

# ============================================================
# LINE 3: VALIDATE THE TOPIC MODEL
# Show 20 random documents from each topic so we can manually
# check whether they really belong to that topic.
# ============================================================
random.seed(42)   # same random documents every run (reproducible)

# Find the top topic for every document
doc_topics = []
for scores in doc_topic_scores:
    doc_topics.append(scores.argmax())

# Group document indexes by topic
docs_per_topic = {}
for topic_number in range(num_topics):
    docs_per_topic[topic_number] = []

for doc_index in range(len(documents)):
    topic_number = doc_topics[doc_index]
    docs_per_topic[topic_number].append(doc_index)

# Save the review to a file so it is easy to read
review_file = open("topic_validation.txt", "w", encoding="utf-8")

print()
print("=" * 50)
print("MANUAL VALIDATION: 20 random documents per topic")
print("=" * 50)

for topic_number in range(num_topics):
    doc_indexes = docs_per_topic[topic_number]

    # Pick up to 20 random documents (a topic may have fewer than 20)
    sample_size = min(20, len(doc_indexes))
    sample = random.sample(doc_indexes, sample_size)

    header = "\nTopic " + str(topic_number) + " (" + str(len(doc_indexes)) + " documents) - showing " + str(sample_size) + ":"
    print(header)
    review_file.write(header + "\n")

    for doc_index in sample:
        # Show the title if available, else first 80 characters
        if "title" in subset.columns:
            preview = str(subset["title"].iloc[doc_index])
        else:
            preview = documents[doc_index][:80]
        line = "  - " + preview
        print(line)
        review_file.write(line + "\n")

review_file.close()
print()
print("Review saved as topic_validation.txt")