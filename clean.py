import pandas
import re

# ---- Cleaning functions ----

def clean_html(text):
    return re.sub(r"<.*?>", "", text)

def clean_special(text):
    return re.sub(r"[^a-zA-Z0-9\s.,]", "", text)

def clean_spaces(text):
    words = text.split()
    return " ".join(words)

def clean_text(text):
    text = clean_html(text)
    text = clean_special(text)
    text = clean_spaces(text)
    return text

# ---- Edge case functions ----

def is_empty(text):
    # True if text is empty or only spaces
    return text.strip() == ""

def is_too_short(text):
    # True if text is shorter than 50 characters
    return len(text.strip()) < 50

def is_too_long(text):
    # Extremely long text: longer than 100000 characters
    return len(text) > 100000

def shorten_if_too_long(text):
    # If text is too long, keep only the first 100000 characters
    if is_too_long(text):
        return text[:100000]
    return text

def is_mostly_english(text):
    # Mixed language check: keep text only if most characters are English letters
    if len(text) == 0:
        return False
    english_letters = re.findall(r"[a-zA-Z]", text)
    ratio = len(english_letters) / len(text)
    return ratio > 0.5

# ---- Load the data ----
if __name__ == "__main__":
    df = pandas.read_csv("articles.csv")
    print("Before cleaning:", len(df), "articles")
    before = len(df)

    cleaned_rows = []
    for index, row in df.iterrows():
        text = str(row["content"])

        # handle extremely long text
        text = shorten_if_too_long(text)

        # clean the text
        text = clean_text(text)

        # skip empty or too short
        if is_empty(text) or is_too_short(text):
            continue

        # skip non-English (mixed language) text
        if not is_mostly_english(text):
            continue

        cleaned_rows.append({"title": row["title"], "content": text})

    cleaned_df = pandas.DataFrame(cleaned_rows)
    after = len(cleaned_df)

    # log how much data was dropped
    dropped = before - after
    percent = round((dropped / before) * 100, 2)
    print("After cleaning:", after, "articles")
    print("Dropped:", dropped, "articles (", percent, "%)")

    cleaned_df.to_csv("cleaned_articles.csv", index=False)
    print("Saved to cleaned_articles.csv")