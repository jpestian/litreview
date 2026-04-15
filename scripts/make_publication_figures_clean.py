#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data/processed/clean_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_WIDTH = 7.0
FIG_HEIGHT = 4.8
DPI = 300
MAX_YEAR = 2025

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10
})

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    return pd.read_csv(INPUT_CSV)

# -----------------------------
# SAVE FIGURE
# -----------------------------
def save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    print(f"Saved {name}")

# -----------------------------
# STYLE
# -----------------------------
def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -----------------------------
# FOOTER
# -----------------------------
def add_footer(fig, df):
    n = len(df)

    if "year" in df.columns:
        temp = df[(df["year"].notna()) & (df["year"] <= MAX_YEAR)]
        if len(temp) > 0:
            footer = f"N = {n} | Years: {int(temp['year'].min())}–{int(temp['year'].max())}"
        else:
            footer = f"N = {n}"
    else:
        footer = f"N = {n}"

    fig.text(0.5, -0.05, footer, ha="center", fontsize=9)

# -----------------------------
# EXTRACT METHODS
# -----------------------------
def extract_methods(df):
    text = df["abstract"].fillna("").str.lower()

    patterns = {
        "Accuracy": r"accuracy",
        "F1": r"f1|f-?score|f score",
        "Precision": r"precision",
        "Recall": r"recall",
        "AUC": r"auc|roc|area under the curve",
        "Cross-validation": r"cross[- ]?validation|k-fold|cv"
    }

    rows = []
    for t in text:
        found = [name for name, pat in patterns.items() if re.search(pat, t)]
        rows.append(found)

    return pd.Series(rows).explode().dropna()

# -----------------------------
# STUDY DESIGN
# -----------------------------
def add_study_design(df):
    text = (df["title"] + " " + df["abstract"]).str.lower()

    def classify(t):
        if "trial" in t:
            return "Clinical Trial"
        if "review" in t:
            return "Review"
        if "model" in t or "algorithm" in t:
            return "Method Development"
        if "cohort" in t or "retrospective" in t:
            return "Observational Study"
        return "Other"

    df["study_design"] = text.apply(classify)
    return df

# -----------------------------
# FIGURE 1: METHODS
# -----------------------------
def fig_methods(df):
    m = extract_methods(df)
    counts = m.value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.barh(counts.index, counts.values)

    ax.set_title("Evaluation Methods")
    ax.set_xlabel("Count")

    clean(ax)
    add_footer(fig, df)
    plt.tight_layout()
    save(fig, "figure_eval_methods")

# -----------------------------
# FIGURE 2: STUDY DESIGN
# -----------------------------
def fig_study_design(df):
    counts = df["study_design"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.barh(counts.index, counts.values)

    ax.set_title("Study Design")
    ax.set_xlabel("Count")

    clean(ax)
    add_footer(fig, df)
    plt.tight_layout()
    save(fig, "figure_study_design")

# -----------------------------
# FIGURE 3: TEMPORAL
# -----------------------------
def fig_temporal_trends(df):
    temp = df.copy()
    temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
    temp = temp[(temp["year"].notna()) & (temp["year"] <= MAX_YEAR)]

    counts = temp.groupby("year").size()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(counts.index, counts.values)

    ax.set_title("Temporal Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.set_xlim(right=MAX_YEAR)

    clean(ax)
    add_footer(fig, temp)
    plt.tight_layout()
    save(fig, "figure_temporal_trends")

# -----------------------------
# FIGURE 4: CROSS
# -----------------------------
def fig_cross(df):
    methods = extract_methods(df)

    methods = methods.reset_index()
    methods.columns = ["row_id", "method"]
    methods["design"] = df.iloc[methods["row_id"]]["study_design"].values

    matrix = pd.crosstab(methods["design"], methods["method"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title("Study Design × Evaluation Methods")
    plt.colorbar(im)

    add_footer(fig, df)
    plt.tight_layout()
    save(fig, "figure_cross")

# -----------------------------
# FIGURE 5: REASONING vs CDS
# -----------------------------
def fig_reasoning_vs_cds(df):

    temp = df.copy()
    temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
    temp = temp[(temp["year"].notna()) & (temp["year"] <= MAX_YEAR)]

    text = (temp["title"] + " " + temp["abstract"]).str.lower()

    reasoning_terms = [
        "clinical reasoning","diagnostic reasoning","clinical inference",
        "medical reasoning","diagnostic model","decision making",
        "clinical decision making","risk stratification"
    ]

    cds_terms = [
        "decision support","clinical decision support","cds",
        "recommendation system","alert system"
    ]

    temp["reasoning"] = text.apply(lambda t: any(k in t for k in reasoning_terms))
    temp["cds"] = text.apply(lambda t: any(k in t for k in cds_terms))

    total = temp.groupby("year").size()
    reasoning = temp[temp["reasoning"]].groupby("year").size()
    cds = temp[temp["cds"]].groupby("year").size()

    years = list(range(int(temp["year"].min()), MAX_YEAR+1))

    total = total.reindex(years, fill_value=0)
    reasoning = reasoning.reindex(years, fill_value=0) / total
    cds = cds.reindex(years, fill_value=0) / total

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(years, reasoning, label="Clinical Reasoning")
    ax.plot(years, cds, label="Decision Support")

    ax.set_title("Clinical Reasoning vs Decision Support")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of Papers")
    ax.set_xlim(right=MAX_YEAR)

    ax.legend()

    clean(ax)
    add_footer(fig, temp)
    plt.tight_layout()
    save(fig, "figure_reasoning_vs_cds")

# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_data()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"].notna()) & (df["year"] <= MAX_YEAR)]

    df = add_study_design(df)

    print("Generating figures...")

    fig_methods(df)
    fig_study_design(df)
    fig_temporal_trends(df)
    fig_cross(df)
    fig_reasoning_vs_cds(df)

    print("Done.")

if __name__ == "__main__":
    main()
