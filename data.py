import wikipediaapi
import pandas
import time

wiki = wikipediaapi.Wikipedia("MyProject", "en")

big_topics = ["Category:Physics", "Category:Chemistry", "Category:Biology",
              "Category:Astronomy", "Category:Mathematics", "Category:Technology",
              "Category:Computer science", "Category:Engineering"]

subcats = []
for topic in big_topics:
    cat = wiki.page(topic)
    for name in cat.categorymembers:
        if name.startswith("Category:"):
            subcats.append(name)

print("Sub-categories :", len(subcats))

names = []
for subcat in subcats:
    cat = wiki.page(subcat)
    for name in cat.categorymembers:
        if ":" not in name:
            names.append(name)

print("Article name:", len(names))

data = []
count = 0
for name in names:
    if count >= 5000:
        break
    try:
        page = wiki.page(name)
        if page.exists():
            data.append({"title": page.title, "content": page.text})
            count = count + 1
            if count % 100 == 0:
                print("done:", count)
                pandas.DataFrame(data).to_csv("articles.csv", index=False)
    except Exception:
        print("Skipped one article, continuing...")
        time.sleep(5)

pandas.DataFrame(data).to_csv("articles.csv", index=False)
print("done! Total articles:", len(data))