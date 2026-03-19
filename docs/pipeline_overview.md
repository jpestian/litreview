# Pipeline Overview

1. Data ingestion (PubMed CSVs)
2. Abstract extraction and cleaning
3. Study design classification (rule-based / LLM-assisted)
4. Clustering and interpretation
5. Figure generation
6. Output tables and publication artifacts

Run:

conda activate nlp-core
cd litreview
python scripts/litrev_pipeline.py
python scripts/make_publication_figures_clean.py
