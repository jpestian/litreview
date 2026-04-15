#!/usr/bin/env python3

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# PATH CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "litreview_output"

TOP_N = 500

EMBED_FILE = OUTPUT_DIR / f"top{TOP_N}_embeddings.npy"
META_FILE = OUTPUT_DIR / "clustered_papers.csv"
LABEL_FILE = OUTPUT_DIR / "cluster_labels.csv"
FIG_FILE = OUTPUT_DIR / "literature_map_labeled.png"

# =========================
# LOAD DATA
# =========================

if not EMBED_FILE.exists():
    raise FileNotFoundError(f"Missing embeddings: {EMBED_FILE}")

if not META_FILE.exists():
    raise FileNotFoundError(f"Missing metadata: {META_FILE}")

if not LABEL_FILE.exists():
    raise FileNotFoundError(f"Missing cluster labels: {LABEL_FILE}")

print("Loading data...")

emb = np.load(EMBED_FILE)
df = pd.read_csv(META_FILE)
labels_df = pd.read_csv(LABEL_FILE, index_col=0)

# =========================
# YEAR RANGE (AUTO TITLE)
# =========================

years = pd.to_numeric(df["year"], errors="coerce").dropna()

if len(years) > 0:
    year_min = int(years.min())
    year_max = int(years.max())
    title = f"Semantic Map of Clinical Decision Support Literature ({year_min}–{year_max})"
else:
    title = "Semantic Map of Clinical Decision Support Literature"

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
# SORT CLUSTERS BY SIZE
# =========================

cluster_sizes = df["cluster"].value_counts()
clusters = cluster_sizes.index.tolist()

# =========================
# PLOT
# =========================

plt.figure(figsize=(11, 9))

for rank, c in enumerate(clusters, start=1):

    name = labels_df.loc[c, "label"] if c in labels_df.index else f"Cluster {c}"

    subset = df[df["cluster"] == c]
    idx = subset.index.to_numpy()
    count = len(subset)

    label = f"{rank}. {name} ({count})"

    plt.scatter(
        coords[idx, 0],
        coords[idx, 1],
        s=80,
        label=label
    )

plt.title(title)
plt.xticks([])
plt.yticks([])

plt.legend(
    title="Research Themes",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(FIG_FILE, dpi=300)

print(f"Saved figure: {FIG_FILE}")

plt.show()
