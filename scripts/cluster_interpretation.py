import pandas as pd
from pathlib import Path
from collections import Counter
import re

OUTPUT_DIR = Path("output")

FILE = OUTPUT_DIR / "clustered_papers.csv"

df = pd.read_csv(FILE)

STOPWORDS = set([
"the","and","for","with","from","using","based","study",
"system","model","models","method","methods","analysis",
"data","approach","results","evaluation","paper","clinical"
])

def tokenize(text):

    text = str(text).lower()

    words = re.findall(r"[a-z]{4,}", text)

    return [w for w in words if w not in STOPWORDS]


print("\nCluster interpretation report\n")

for c in sorted(df.cluster.unique()):

    sub = df[df.cluster == c]

    tokens = []

    for _, r in sub.iterrows():

        tokens += tokenize(r["title"])
        tokens += tokenize(r.get("abstract",""))

    counts = Counter(tokens)

    top = counts.most_common(12)

    print("\nCluster", c)
    print("Top keywords:")

    for w,n in top:
        print(f"{w:15} {n}")

    print("\nExample papers:")

    examples = sub.sort_values(
        "cited_by_count",
        ascending=False
    ).head(5)

    for t in examples.title:
        print(" -",t)

    print("\n" + "-"*60)
