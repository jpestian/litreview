python - << 'EOF'
import pandas as pd

# Load clustered data (missing abstracts)
clustered = pd.read_csv("results/litreview_output/clustered_papers_with_designs.csv")

# Load raw PubMed data (has abstracts)
raw = pd.read_csv("data/pubmed03142024.csv")

# Normalize join key
clustered["title"] = clustered["title"].str.strip().str.lower()
raw["title"] = raw["title"].str.strip().str.lower()

# Merge abstracts back
merged = clustered.merge(
    raw[["title", "abstract"]],
    on="title",
    how="left"
)

# Save fixed dataset
merged.to_csv(
    "results/litreview_output/clustered_with_abstracts.csv",
    index=False
)

print("Saved clustered_with_abstracts.csv")
print("Abstract count:", merged["abstract"].notna().sum())
EOF
