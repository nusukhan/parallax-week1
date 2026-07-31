import pandas

df = pandas.read_csv("articles.csv")

print("=== DATA QUALITY REPORT ===")
print("Total records:", len(df))
print("Columns:", list(df.columns))
print()
print("Missing values per column:")
print(df.isnull().sum())
print()
missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
print("Missing fields percentage:", round(missing_percent, 2), "%")
print()
duplicates = df.duplicated(subset=["title"]).sum()
print("Duplicate titles:", duplicates)
print("Duplicate rate:", round((duplicates / len(df)) * 100, 2), "%")
print()
print("Empty content rows:", (df["content"].str.strip() == "").sum())
print("Average content length:", int(df["content"].str.len().mean()), "characters")
print()
print("--- Encoding Check ---")
bad_chars = ["â€™", "â€œ", "â€", "Ã©", "Ã¨", "\ufffd"]
bad_count = 0
for char in bad_chars:
    count = df["content"].str.contains(char, regex=False).sum()
    if count > 0:
        print("Found bad character", char, "in", count, "rows")
        bad_count = bad_count + count
if bad_count == 0:
    print("No encoding issues found - all text is clean!")
else:
    print("Total rows with encoding issues:", bad_count)