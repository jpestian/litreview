# Reproducibility and Execution Log

This document captures the full terminal-level workflow used to construct, debug, and validate the literature review pipeline. It reflects actual execution rather than an idealized sequence.

The goal is reproducibility through transparency: directory structure, environment configuration, command execution, and failure handling are all included.

---

## 1. Environment Setup

Activate working environment:

```bash
conda activate nlp-core
```

Locate environments:

```bash
conda info --envs
```

Export environment:

```bash
conda env export > environment/environment.yml
```

---

## 2. Environment Cleanup (Required for Portability)

The exported environment contains machine-specific dependencies and Python 3.13 bindings that are not portable.

Edit the environment file:

```bash
vi environment/environment.yml
```

Remove all build-specific lines:

```bash
:%g/=py313/d
```

Replace with a minimal, portable environment:

```yaml
name: litreview-env

channels:
  - defaults
  - conda-forge

dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - ipython
  - jupyterlab
  - pip

  - pip:
      - sentence-transformers
      - transformers
      - torch
      - hdbscan
      - umap-learn
```

---

## 3. Environment Validation

```bash
conda env remove -n litreview-test
conda env create -f environment/environment.yml -n litreview-test
```

---

## 4. Project Navigation

```bash
cd ~/projects/litreview
tree -L 2
```

Expected structure:

```
data/
outputs/
scripts/
docs/
notebooks/
queries/
environment/
```

---

## 5. Data Organization

Create directory structure:

```bash
mkdir -p data/raw data/processed
```

Move raw data:

```bash
mv data/pubmed*.csv data/raw/
mv data/csv-* data/raw/
mv data/*.txt data/raw/
```

Move processed data:

```bash
mv data/raw/pubmed_with_abstracts.csv data/processed/
mv data/clean_dataset.csv data/processed/ 2>/dev/null
```

Verify:

```bash
tree data/
```

---

## 6. Outputs Organization

Create output structure:

```bash
mkdir -p outputs/{figures,pdf,tables,embeddings}
```

Move figures:

```bash
mv figures/*.png outputs/figures/
mv figures/*.pdf outputs/pdf/
rmdir figures
```

Move results:

```bash
mv results/* outputs/tables/
rm results/.gitkeep
rmdir results
```

Separate mixed artifacts:

```bash
mv outputs/tables/litreview_output/*.png outputs/figures/ 2>/dev/null
mv outputs/tables/litreview_output/*.pdf outputs/pdf/ 2>/dev/null
mv outputs/tables/litreview_output/*.npy outputs/embeddings/
```

---

## 7. Script Execution

Run full pipeline:

```bash
python scripts/litrev_pipeline.py
```

Generate publication figures:

```bash
python scripts/make_publication_figures_clean.py
```

Run test pipeline:

```bash
python scripts/test_top10_pipeline.py
```

---

## 8. Script Structure

```
scripts/
├── classify_study_designs.py
├── cluster_interpretation.py
├── litrev_pipeline.py
├── make_publication_figures_clean.py
├── mergeabstracts.py
├── sort_csv_to_bib_test.py
├── test_top10_pipeline.py
└── visualize_clusters.py
```

---

## 9. Debugging and Failure Modes

### Missing file

```bash
mv: cannot stat 'file'
```

Resolution:

```bash
find . -iname "*filename*"
```

---

### Directory not empty

```bash
rmdir: Directory not empty
```

Resolution:

```bash
ls -la directory/
rm directory/.gitkeep
rmdir directory
```

---

### Script path errors

```bash
python: can't open file
```

Resolution:

- Ensure execution from project root  
- Avoid nested relative paths  

---

### Environment errors

```bash
ERROR: No matching distribution found for torch==2.10.0+cpu
```

Resolution:

- remove `+cpu`  
- avoid strict version pinning  
- use Python 3.10  

---

## 10. Git Integration

```bash
git status
git add .
git commit -m "Finalize reproducible pipeline"
git push origin main
```

---

## 11. Observations

- Most failures were path-related rather than algorithmic  
- Environment export requires manual cleanup  
- Strict version pinning reduces portability  
- Separation of raw vs processed data simplifies reasoning  
- Execution from project root is required for consistency  

---

## 12. Minimal Reproducible Workflow

```bash
conda activate nlp-core
cd ~/projects/litreview

python scripts/litrev_pipeline.py
python scripts/make_publication_figures_clean.py
```

---

## 13. Final Structure

```
data/
  raw/
  processed/

outputs/
  figures/
  pdf/
  tables/
  embeddings/

scripts/
docs/
environment/
```

---

## 14. Summary

This repository implements a reproducible pipeline for analyzing artificial intelligence and biomedical informatics literature using abstract-level text.

The workflow includes:

- structured data ingestion  
- rule-based and embedding-based classification  
- clustering and interpretation  
- figure generation for publication  

All results are reproducible using the scripts and environment defined in this repository.

B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
B
The execution record above reflects the actual process used to generate the results, including intermediate failures and their resolution.
