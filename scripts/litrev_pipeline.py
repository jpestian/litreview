#!/usr/bin/env python3

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

# ------------------------------------------------
# PATH CONFIG
# ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = sorted(
    SCRIPT_DIR.glob("*.csv"),
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

INPUT_FILE = csv_files[0]

TOP_FOR_CLUSTERING = 500

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------

def clean(v):
    if not v:
        return ""
    return " ".join(str(v).split()).strip()

def norm_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/","",clean(d)).lower()

def norm_pmid(p):
    return re.sub(r"\D","",clean(p))

def chunk(lst,n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

# ------------------------------------------------
# STUDY DESIGN CLASSIFIER
# ------------------------------------------------

def classify_design(text):

    t=text.lower()

    if "randomized controlled trial" in t:
        return "RCT"

    if "cluster randomized" in t:
        return "Cluster RCT"

    if "diagnostic accuracy" in t:
        return "Diagnostic Accuracy"

    if "retrospective" in t:
        return "Retrospective Cohort"

    if "prospective" in t:
        return "Prospective Cohort"

    if "implementation" in t:
        return "Implementation"

    if "simulation" in t or "usability" in t:
        return "Simulation"

    if "systematic review" in t or "meta-analysis" in t:
        return "Review"

    return "Other"

# ------------------------------------------------
# OPENALEX CITATIONS
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

    async with session.get(url) as r:

        data=await r.json()

    works=data.get("results",[])

    citation_map={}

    for w in works:

        doi_raw=w.get("doi","")

        if doi_raw:
            citation_map[norm_doi(doi_raw)]=w.get("cited_by_count",0)

    for p in batch:

        p["cited_by_count"]=citation_map.get(
            norm_doi(p["doi"]),0
        )

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
# PUBMED ABSTRACTS
# ------------------------------------------------

PUBMED_EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

async def get_pubmed_abstracts(pmids):

    params={
        "db":"pubmed",
        "id":",".join(pmids),
        "retmode":"xml"
    }

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

        parts=[]

        for a in art.findall(".//AbstractText"):
            parts.append("".join(a.itertext()))

        abstracts[pmid]=" ".join(parts)

    return abstracts

# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

def embed(texts):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return emb

# ------------------------------------------------
# CLUSTERING
# ------------------------------------------------

def cluster(emb):

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )

    coords = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5
    )

    labels = clusterer.fit_predict(coords)

    # Fix noise cluster
    if -1 in labels:

        mask = labels != -1

        clf = NearestCentroid()

        clf.fit(coords[mask], labels[mask])

        noise = np.where(labels==-1)[0]

        labels[noise]=clf.predict(coords[noise])

    return labels, coords

# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    rows=list(csv.DictReader(open(INPUT_FILE)))

    papers=[]

    for r in rows:

        papers.append({
            "pmid":norm_pmid(r.get("PMID")),
            "title":clean(r.get("Title")),
            "year":clean(r.get("Publication Year")),
            "doi":norm_doi(r.get("DOI")),
            "cited_by_count":0
        })

    print("Fetching citations...")

    papers=asyncio.run(get_citations(papers))

    papers=sorted(
        papers,
        key=lambda x:x["cited_by_count"],
        reverse=True
    )

    top500=papers[:TOP_FOR_CLUSTERING]

    print("Fetching abstracts...")

    pmids=[p["pmid"] for p in top500]

    abstracts=asyncio.run(get_pubmed_abstracts(pmids))

    for p in top500:
        p["abstract"]=abstracts.get(p["pmid"],"")

    texts=[p["title"]+" "+p["abstract"] for p in top500]

    print("Embedding papers...")

    emb=embed(texts)

    print("Clustering...")

    labels,coords=cluster(emb)

    df=pd.DataFrame(top500)

    df["cluster"]=labels

    df.to_csv(
        OUTPUT_DIR/"clustered_papers.csv",
        index=False
    )

    # ------------------------------------------------
    # SHOW CLUSTERS
    # ------------------------------------------------

    print("\nTop titles per cluster\n")

    for c in sorted(df.cluster.unique()):

        print("\nCluster",c)

        sub=df[df.cluster==c].head(10)

        for t in sub.title:
            print(" -",t)

    # ------------------------------------------------
    # STUDY DESIGN TABLE
    # ------------------------------------------------

    df["study_design"]=df["abstract"].apply(classify_design)

    table=pd.crosstab(
        df["cluster"],
        df["study_design"]
    )

    table.to_csv(
        OUTPUT_DIR/"cluster_study_design_table.csv"
    )

    print("\nStudy Design Table:\n")
    print(table)

    # ------------------------------------------------
    # LITERATURE MAP
    # ------------------------------------------------

    plt.figure(figsize=(10,8))

    plt.scatter(
        coords[:,0],
        coords[:,1],
        c=df.cluster,
        cmap="tab10",
        s=60
    )

    plt.title("Semantic Map of Clinical Decision Support Literature")

    plt.xticks([])
    plt.yticks([])

    plt.savefig(
        OUTPUT_DIR/"literature_map.png",
        dpi=300
    )

    plt.savefig(
        OUTPUT_DIR/"literature_map.pdf"
    )

    print("\nOutputs written to:",OUTPUT_DIR)

# ------------------------------------------------

if __name__=="__main__":
    main()
