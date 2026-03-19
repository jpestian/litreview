#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# PATH CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "litreview_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_DIR / "clustered_papers.csv"

TABLE_FILE = OUTPUT_DIR / "cluster_study_design_simple.csv"
PROP_FILE = OUTPUT_DIR / "cluster_study_design_simple_proportions.csv"
FIG_FILE = OUTPUT_DIR / "final_publication_figure.png"

# =========================
# LOAD DATA
# =========================

print("Loading clustered literature...")
print("INPUT_FILE:", INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")

# =========================
# TEXT CLEANING
# =========================

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# STUDY DESIGN CLASSIFIER
# =========================

def classify_design(title, abstract):

    text = normalize(title + " " + abstract)

    # Clinical studies
    if any(x in text for x in [
        "randomized", "prospective", "retrospective",
        "clinical trial", "cohort"
    ]):
        return "Clinical Study"

    # Model evaluation
    if any(x in text for x in [
        "auc", "roc", "accuracy", "validation",
        "performance", "sensitivity", "specificity"
    ]):
        return "Model Evaluation"

    # Model development
    if any(x in text for x in [
        "machine learning", "deep learning",
        "neural network", "model", "algorithm"
    ]):
        return "Model Development"

    return "Model Development"

# =========================
# APPLY CLASSIFICATION
# =========================

print("Classifying study designs...")

df["study_design"] = [
    classify_design(t, a)
    for t, a in zip(df["title"], df["abstract"])
]

# =========================
# TABLES
# =========================

count_table = pd.crosstab(
    df["cluster"],
    df["study_design"]
)

prop_table = pd.crosstab(
    df["cluster"],
    df["study_design"],
    normalize="index"
).round(2)

print("\nCounts:\n", count_table)
print("\nProportions:\n", prop_table)

# =========================
# GLOBAL SUMMARY
# =========================

print("\n=== GLOBAL SUMMARY ===\n")

avg = prop_table.mean().round(2)
min_vals = prop_table.min().round(2)
max_vals = prop_table.max().round(2)

print("Average:\n", avg)
print("\nMin:\n", min_vals)
print("\nMax:\n", max_vals)

# =========================
# CLUSTER RANKING
# =========================

print("\n=== CLUSTER RANKING (Clinical Maturity) ===\n")

ranked = prop_table.sort_values("Clinical Study", ascending=False)

for i, (cluster, row) in enumerate(ranked.iterrows(), start=1):
    print(
        f"{i:2d}. Cluster {cluster} | "
        f"Clinical={row['Clinical Study']:.2f}, "
        f"Eval={row['Model Evaluation']:.2f}, "
        f"Dev={row['Model Development']:.2f}"
    )

# =========================
# FINAL FIGURE
# =========================

sorted_df = prop_table.sort_values("Clinical Study", ascending=False)

plt.figure(figsize=(10, 8))

im = plt.imshow(sorted_df.values, aspect='auto', cmap='viridis')

cbar = plt.colorbar(im)
cbar.set_label("Proportion")

plt.xticks(range(len(sorted_df.columns)), sorted_df.columns, rotation=45, ha='right')
plt.yticks(range(len(sorted_df.index)), sorted_df.index)

plt.title(
    "Study Design Distribution Across Clinical AI Clusters\n"
    f"(Development={avg['Model Development']:.2f}, "
    f"Evaluation={avg['Model Evaluation']:.2f}, "
    f"Clinical={avg['Clinical Study']:.2f})"
)

plt.tight_layout()
plt.savefig(FIG_FILE, dpi=300)

print("\nSaved figure:", FIG_FILE)

plt.show()

# =========================
# SAVE OUTPUTS
# =========================

count_table.to_csv(TABLE_FILE)
prop_table.to_csv(PROP_FILE)

print("\nSaved tables:")
print(TABLE_FILE)
print(PROP_FILE)
