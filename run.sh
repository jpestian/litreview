#!/usr/bin/env bash

set -e

echo "----------------------------------"
echo "LitReview Pipeline بدء"
echo "----------------------------------"

# Create environment if not exists
if ! conda env list | grep -q litreview-env; then
  echo "Creating environment..."
  conda env create -f environment/environment.yml -n litreview-env
fi

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate litreview-env

echo "Running pipeline..."
python scripts/litrev_pipeline.py

echo "Generating figures..."
python scripts/make_publication_figures_clean.py

echo "----------------------------------"
echo "Done. Outputs in outputs/"
echo "----------------------------------"
