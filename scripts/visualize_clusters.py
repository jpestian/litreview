#!/usr/bin/env python3

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# FILE PATHS
# =========================

OUTPUT_DIR = Path("output")

EMBED_FILE = OUTPUT_DIR / "top100_embeddings.npy"
META_FILE = OUTPUT_DIR / "literature_clusters.csv"

FIG_FILE = OUTPUT_DIR / "literature_map.png"

# =========================
# CLUSTER NAMES
# =========================

CLUSTER_NAMES = {
0: "Clinical Decision Support Systems",
1: "Machine Learning Diagnostic Models",
2: "Probabilistic Diagnostic Reasoning",
3: "Implementation and Workflow Integration",
4: "Evaluation Methods and Clinical Trials",
5: "Human Factors and Clinical Reasoning"
}

# =========================
# LOAD DATA
# =========================

print("Loading embeddings...")
emb = np.load(EMBED_FILE)

print("Loading metadata...")
df = pd.read_csv(META_FILE)

df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# =========================
# YEAR RANGE
# =========================

year_min = int(df["year"].min())
year_max = int(df["year"].max())

title = f"Semantic Map of Clinical Decision Support Literature ({year_min}-{year_max})"

# =========================
# UMAP PROJECTION
# =========================

print("Running UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.15,
    random_state=42
)

coords = reducer.fit_transform(emb)

# =========================
# PLOT
# =========================

plt.figure(figsize=(11,9))

clusters = sorted(df["cluster"].unique())

for c in clusters:

    name = CLUSTER_NAMES.get(c, f"Cluster {c}")

    subset = df[df["cluster"] == c]

    count = len(subset)

    label = f"{name} ({count})"

    idx = subset.index

    plt.scatter(
        coords[idx,0],
        coords[idx,1],
        s=80,
        label=label
    )

plt.title(title)

# remove meaningless axes
plt.xticks([])
plt.yticks([])

plt.legend(
    title="Research Themes",
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(FIG_FILE, dpi=300)

print("Saved:", FIG_FILE)

plt.show()
