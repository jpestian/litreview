#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path

# =========================
# FILE PATHS
# =========================

OUTPUT_DIR = Path("output")

INPUT_FILE = OUTPUT_DIR / "literature_clusters.csv"

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

print("Loading clustered literature...")

df = pd.read_csv(INPUT_FILE)

# ensure text columns are safe
df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")

# add cluster names
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# =========================
# TEXT UTILITIES
# =========================

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text, phrases):
    return any(p in text for p in phrases)

# =========================
# STUDY DESIGN CLASSIFIER
# =========================

def classify_design(title, abstract):

    text = normalize(title + " " + abstract)

    # systematic reviews
    if contains_any(text, [
        "systematic review",
        "scoping review",
        "meta-analysis",
        "meta analysis",
        "literature review",
        "review of"
    ]):
        return "Review Paper"

    # trial types
    if contains_any(text, ["stepped wedge", "stepped-wedge"]):
        return "Stepped Wedge Trial"

    if contains_any(text, [
        "cluster randomized",
        "cluster randomised",
        "cluster-randomized"
    ]):
        return "Cluster Randomized Trial"

    if contains_any(text, [
        "randomized controlled trial",
        "randomised controlled trial",
        "randomized trial",
        "randomised trial",
        "rct"
    ]):
        return "Randomized Controlled Trial"

    # diagnostic accuracy
    if contains_any(text, [
        "diagnostic accuracy",
        "sensitivity and specificity",
        "receiver operating characteristic",
        "roc curve",
        "area under the curve",
        "auc",
        "validation study"
    ]):
        return "Diagnostic Accuracy Study"

    # simulation / human factors
    if contains_any(text, [
        "simulation study",
        "usability study",
        "human factors",
        "user study",
        "focus group",
        "interview study",
        "mixed methods",
        "mixed-methods"
    ]):
        return "Simulation / Human Factors Study"

    # implementation
    if contains_any(text, [
        "implementation study",
        "implementation trial",
        "feasibility study",
        "pilot study",
        "workflow integration",
        "deployment"
    ]):
        return "Implementation Study"

    # retrospective cohort
    if contains_any(text, [
        "retrospective cohort",
        "retrospective study",
        "retrospective analysis"
    ]):
        return "Retrospective Cohort Study"

    # prospective cohort
    if contains_any(text, [
        "prospective cohort",
        "prospective study",
        "prospective observational"
    ]):
        return "Prospective Cohort Study"

    # observational
    if contains_any(text, [
        "observational study",
        "cross sectional",
        "cross-sectional",
        "survey study"
    ]):
        return "Observational Study"

    if "trial" in text:
        return "Trial (Unspecified)"

    return "Other"


# =========================
# CLASSIFY PAPERS
# =========================

print("Classifying study designs...")

df["study_design"] = [
    classify_design(t, a)
    for t, a in zip(df["title"], df["abstract"])
]

# =========================
# SUMMARY TABLE
# =========================

table = pd.crosstab(
    df["cluster_name"],
    df["study_design"]
)

print("\nStudy Design Table:\n")
print(table)

# =========================
# SAVE OUTPUTS
# =========================

table_file = OUTPUT_DIR / "cluster_study_design_table.csv"
data_file = OUTPUT_DIR / "literature_clusters_with_designs.csv"

table.to_csv(table_file)
df.to_csv(data_file, index=False)

print("\nSaved files:")
print(table_file)
print(data_file)
