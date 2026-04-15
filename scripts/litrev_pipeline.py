#!/usr/bin/env python

import asyncio
import aiohttp
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data/raw"
OUTPUT_DIR = BASE_DIR / "outputs/tables/litreview_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

INPUT_FILE = csv_files[0]
print(f"Using input file: {INPUT_FILE}")

TOP_FOR_CLUSTERING = 500

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------

def clean(v):
    return " ".join(str(v).split()).strip() if v else ""

def norm_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/","",clean(d)).lower()

def norm_pmid(p):
    return re.sub(r"\D","",clean(p))

def chunk(lst,n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

# ------------------------------------------------
# OPENALEX
# ------------------------------------------------

OPENALEX_BASE="https://api.openalex.org/works"

async def openalex_batch(session,batch):

    filters=[]
    for p in batch:
        doi=norm_doi(p["doi"])
        pmid=norm_pmid(p["pmid"])
        if doi:
            filters.append(f"doi:{doi}")
        elif pmid:
            filters.append(f"pmid:{pmid}")

    if not filters:
        return batch

    url=f"{OPENALEX_BASE}?filter={'|'.join(filters)}&per-page=200"

    headers = {"Accept-Encoding": "gzip, deflate"}

    async with session.get(url, headers=headers) as r:
        data=await r.json()

    works=data.get("results",[])
    citation_map={}

    for w in works:
        doi_raw=w.get("doi","")
        if doi_raw:
            citation_map[norm_doi(doi_raw)]=w.get("cited_by_count",0)

    for p in batch:
        p["cited_by_count"]=citation_map.get(norm_doi(p["doi"]),0)

    return batch

async def get_citations(papers):

    async with aiohttp.ClientSession() as session:

        tasks=[]
        for b in chunk(papers,50):
            tasks.append(openalex_batch(session,b))

        results=[]
        for coro in asyncio.as_completed(tasks):
            results.extend(await coro)

        return results

# ------------------------------------------------
# PUBMED
# ------------------------------------------------

PUBMED_EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

async def get_pubmed_abstracts(pmids):

    params={"db":"pubmed","id":",".join(pmids),"retmode":"xml"}

    async with aiohttp.ClientSession() as session:
        async with session.get(PUBMED_EFETCH,params=params) as r:
            xml=await r.text()

    abstracts={}

    try:
        root=ET.fromstring(xml)
    except:
        return abstracts

    for art in root.findall(".//PubmedArticle"):
        pmid=art.findtext(".//PMID")
        parts=[ "".join(a.itertext()) for a in art.findall(".//AbstractText") ]
        abstracts[pmid]=" ".join(parts)

    return abstracts

# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

def embed(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# ------------------------------------------------
# CLUSTERING
# ------------------------------------------------

def cluster(emb):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(coords)

    if -1 in labels:
        mask = labels != -1
        clf = NearestCentroid()
        clf.fit(coords[mask], labels[mask])
        noise = np.where(labels==-1)[0]
        labels[noise]=clf.predict(coords[noise])

    return labels, coords

# ------------------------------------------------
# AUTO LABELING
# ------------------------------------------------

def generate_cluster_labels(df, n_terms=5):

    labels = {}

    for c in sorted(df["cluster"].unique()):

        subset = df[df["cluster"] == c]
        texts = (subset["title"] + " " + subset["abstract"]).fillna("")

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(texts)

        scores = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        top_terms = [terms[i] for i in scores.argsort()[::-1][:n_terms]]

        labels[c] = ", ".join(top_terms)

    return labels

# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    rows=list(csv.DictReader(open(INPUT_FILE)))

    papers=[{
        "pmid":norm_pmid(r.get("PMID")),
        "title":clean(r.get("Title")),
        "year":clean(r.get("Publication Year")),
        "doi":norm_doi(r.get("DOI")),
        "cited_by_count":0
    } for r in rows]

    print("Fetching citations...")
    papers=asyncio.run(get_citations(papers))

    papers=sorted(papers,key=lambda x:x["cited_by_count"],reverse=True)
    top500=papers[:TOP_FOR_CLUSTERING]

    print("Fetching abstracts...")
    pmids=[p["pmid"] for p in top500]
    abstracts=asyncio.run(get_pubmed_abstracts(pmids))

    for p in top500:
        p["abstract"]=abstracts.get(p["pmid"],"")

    texts=[p["title"]+" "+p["abstract"] for p in top500]

    print("Embedding...")
    emb=embed(texts)

    np.save(OUTPUT_DIR / f"top{TOP_FOR_CLUSTERING}_embeddings.npy", emb)

    print("Clustering...")
    labels,coords=cluster(emb)

    df=pd.DataFrame(top500)
    df["cluster"]=labels

    # AUTO LABELS
    cluster_labels = generate_cluster_labels(df)
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    pd.DataFrame.from_dict(cluster_labels, orient="index", columns=["label"]) \
        .to_csv(OUTPUT_DIR / "cluster_labels.csv")

    df.to_csv(OUTPUT_DIR/"clustered_papers.csv",index=False)

    print("\nOutputs written to:",OUTPUT_DIR)

if __name__=="__main__":
    main()
