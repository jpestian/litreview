import csv
import requests
import time

INPUT_FILE = "/home/jpestian/Downloads/csv-DecisionSu-set.csv"
OUTPUT_FILE = "/home/jpestian/Downloads/test_sorted_by_citations.bib"

TEST_ROWS = 5

OPENALEX_URL = "https://api.openalex.org/works"

session = requests.Session()


def get_citations(doi=None, pmid=None):

    try:
        if doi:
            doi = doi.replace("https://doi.org/", "")
            r = session.get(f"{OPENALEX_URL}?filter=doi:{doi}&per-page=1")
        elif pmid:
            r = session.get(f"{OPENALEX_URL}?filter=pmid:{pmid}&per-page=1")
        else:
            return 0

        data = r.json()

        if data["results"]:
            return data["results"][0]["cited_by_count"]

    except:
        pass

    return 0


papers = []

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        if i >= TEST_ROWS:
            break

        title = row.get("Title", "")
        authors = row.get("Authors", "")
        year = row.get("Publication Year", "")
        doi = row.get("DOI", "")
        pmid = row.get("PMID", "")

        print("Checking:", title[:60])

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


with open(OUTPUT_FILE, "w") as f:

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


print("\nDone.")
print("Test BibTeX file:", OUTPUT_FILE)

print("\nSorted results:")
for p in papers:
    print(p["citations"], "-", p["title"])
