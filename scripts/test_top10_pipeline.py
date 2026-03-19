import csv
import requests
import time
import xml.etree.ElementTree as ET


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "csv-DecisionSu-set.csv"
OUTPUT_DIR = "results/test_pipeline_output"


OPENALEX = "https://api.openalex.org/works"
PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

TEST_ROWS = 10

session = requests.Session()


def get_citations(doi=None, pmid=None):

    try:
        if doi:
            doi = doi.replace("https://doi.org/", "")
            r = session.get(f"{OPENALEX}?filter=doi:{doi}&per-page=1")
        elif pmid:
            r = session.get(f"{OPENALEX}?filter=pmid:{pmid}&per-page=1")
        else:
            return 0

        data = r.json()

        if data["results"]:
            return data["results"][0]["cited_by_count"]

    except:
        pass

    return 0


def get_abstract(pmid):

    try:
        r = session.get(PUBMED, params={
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        })

        root = ET.fromstring(r.text)

        texts = []

        for t in root.findall(".//AbstractText"):
            texts.append("".join(t.itertext()))

        return " ".join(texts)

    except:
        return ""


papers = []

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        if i >= TEST_ROWS:
            break

        title = row["Title"]
        authors = row["Authors"]
        year = row["Publication Year"]
        doi = row["DOI"]
        pmid = row["PMID"]

        print("Processing:", title[:60])

        citations = get_citations(doi, pmid)

        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "pmid": pmid,
            "citations": citations
        })

        time.sleep(0.2)


papers = sorted(papers, key=lambda x: x["citations"], reverse=True)


import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Write BibTeX

with open(f"{OUTPUT_DIR}/top10_test.bib", "w") as f:

    for i, p in enumerate(papers):

        key = f"paper{i+1}"

        f.write(f"""
@article{{{key},
  title={{ {p['title']} }},
  author={{ {p['authors']} }},
  year={{ {p['year']} }},
  doi={{ {p['doi']} }},
  url={{ https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/ }},
  note={{Citations: {p['citations']}}}
}}
""")


# Fetch abstracts

rows = []

for p in papers:

    abstract = get_abstract(p["pmid"])

    rows.append({
        "title": p["title"],
        "citations": p["citations"],
        "abstract": abstract
    })

    time.sleep(0.3)


# Write CSV

import pandas as pd

df = pd.DataFrame(rows)
df.to_csv(f"{OUTPUT_DIR}/top10_abstracts.csv", index=False)


# Write ChatGPT summary file

with open(f"{OUTPUT_DIR}/top10_for_chatgpt.txt", "w") as f:

    f.write("""
Summarize these abstracts according to three types of systems:

1. Clinical reasoning support
2. Clinical decision support
3. Probabilistic diagnostic systems

Explain the differences and overlaps.
""")

    for r in rows:

        f.write("\n\n---------------------------------\n")
        f.write(r["title"])
        f.write("\n")
        f.write(r["abstract"])


print("\nTEST COMPLETE")

print("Files created:")
print(f"{OUTPUT_DIR}/top10_test.bib")
print(f"{OUTPUT_DIR}/top10_abstracts.csv")
print(f"{OUTPUT_DIR}/top10_for_chatgpt.txt")
