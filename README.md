# Literature Review Pipeline

## Overview

This repository implements a reproducible pipeline for analyzing and clustering the artificial intelligence and biomedical informatics literature using abstract-level text.

The workflow includes:

- ingestion of PubMed-derived datasets  
- abstract extraction and preprocessing  
- transformer-based embeddings (Sentence-BERT)  
- clustering using UMAP and HDBSCAN  
- rule-based classification of study design and evaluation methods  
- automated generation of tables and publication-ready figures  

---

## Structure

```
data/
  raw/          # input datasets (PubMed CSV files)
  processed/    # cleaned and derived datasets

outputs/
  figures/      # PNG figures
  pdf/          # publication-ready figures
  tables/       # CSV outputs
  embeddings/   # numerical representations

scripts/        # pipeline and analysis code
docs/           # reproducibility and execution log
notebooks/      # exploratory analysis (optional)
environment/    # environment configuration
```

---

## Run the Pipeline

Clone and execute:

```bash
git clone <repo>
cd litreview
./run.sh
```

This will:

1. create the environment  
2. run the full pipeline  
3. generate all outputs  

Results are written to:

```
outputs/
```

## Environment Setup (Manual Option)

If you prefer to set up the environment manually:

```bash
conda env create -f environment/environment.yml -n litreview-env
conda activate litreview-env
```

Verify installation:

```bash
python -c "import torch, sklearn, sentence_transformers, aiohttp; print('OK')"
```
## Requirements

- Conda (Miniconda or Anaconda)
- Python 3.10 (handled automatically by the environment)

## Reproducibility

A full execution log is provided:

```
docs/reproducibility_and_execution.md
```

This documents the exact commands used to generate the results, including debugging steps.

---

## Methods Overview

- Abstracts are embedded using Sentence-BERT (*all-MiniLM-L6-v2*)  
- Dimensionality reduction is performed using UMAP  
- Clustering is performed using HDBSCAN  
- Study design and evaluation methods are extracted using rule-based methods  

---

## Notes

- Analysis is based on abstract-level text  
- Results reflect structural patterns rather than full-text interpretation  
- The pipeline is designed for consistency and scalability  

---

## Code and Data Availability

All code, environment configuration, and execution steps required to reproduce the analysis are available in this repository.

The analysis can be reproduced using a single command after cloning the repository.

---

## Contact
John Pestian,PhD
john.pestian@cchmc.org
