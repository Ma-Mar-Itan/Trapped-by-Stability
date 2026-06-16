# Trapped by Stability

### A Falsification Framework for Acquisition-Driven Radiomic Phenotypes

Developed as part of:

**Quality Assurance of Medical Images**


---

## Overview

This repository contains the full, reproducible analysis pipeline used to
investigate acquisition confounding in multi-vendor breast-MRI radiomic
phenotype discovery.

Using a cohort of **922 breast-MRI patients** (GE and Siemens scanners), the
study demonstrates that:

* highly stable unsupervised radiomic clusters can still be acquisition-driven
  rather than biologically meaningful,
* conventional clustering-stability metrics alone are insufficient for
  validating radiomic phenotypes,
* harmonization methods can fundamentally alter cluster identity and downstream
  prognostic associations.

The core methodological conclusion:

> **Cluster stability and cluster biological identity are independent
> properties.**

## Main research question

Can a radiomic phenotype remain statistically stable while failing biological
validity checks due to acquisition confounding? The pipeline evaluates this via
multi-condition phenotype discovery, harmonization sensitivity testing,
acquisition-variance partitioning, within-vendor validation, and vendor-clean
phenotype scoring. See [`docs/scientific_workflow.md`](docs/scientific_workflow.md).

## Project structure

```
.
├── data/
│   ├── raw/          # inputs: Clinical, Imaging_Features, Annotation_Boxes
│   ├── interim/      # merged_dataset.csv
│   ├── processed/    # z-scored matrix, PCA scores, harmonized matrices
│   └── results/      # cluster labels, statistical tables, figures
├── src/radiomics_stability/
│   ├── config.py             # all seeds / hyperparameters / column names
│   ├── paths.py              # machine-independent paths
│   ├── preprocessing/        # merge_data, prepare_radiomics
│   ├── clustering/           # pca, kmeans, stability
│   ├── statistics/           # biology_analysis, manufacturer_analysis, survival
│   ├── harmonization/        # residualization, combat, validation
│   └── visualization/        # plots
├── scripts/                  # CLI entry points (run_full_pipeline.py, ...)
├── docs/                     # scientific_workflow.md, migration_log.md
├── requirements.txt
└── pyproject.toml
```

## Installation

Python ≥ 3.10 is required. Library versions matter for exact reproduction
(KMeans and randomized PCA are version-sensitive) — install from
`requirements.txt`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
pip install -e .            # optional: installs the package for `import radiomics_stability`
```

## Running the pipeline

Place the three raw CSVs in `data/raw/`, then regenerate **all** results from a
fresh clone with one command:

```bash
python scripts/run_full_pipeline.py
```

Or run individual stages (each depends on the previous):

```bash
python scripts/run_preprocessing.py     # merge raw → z-scored feature matrix
python scripts/run_clustering.py         # PCA, KMeans phenotype, bootstrap stability
python scripts/run_statistics.py         # biological + manufacturer analysis
python scripts/run_survival.py           # HR+/HER2+ recurrence-free survival
python scripts/run_harmonization.py      # residualization, ComBat, validation
```

If `neuroCombat` is unavailable, run everything except harmonization with
`python scripts/run_full_pipeline.py --skip-harmonization`.

Outputs are written under `data/interim`, `data/processed`, and `data/results`.

## Configuration

All tunable scientific constants live in
[`src/radiomics_stability/config.py`](src/radiomics_stability/config.py) —
including the global seed:

```python
RANDOM_SEED = 42
```

Paths can be overridden with the `RADSTAB_PROJECT_ROOT` / `RADSTAB_DATA_DIR`
environment variables.

## Reproducibility

The pipeline is deterministic. This refactor was verified to reproduce the
original committed outputs byte-for-byte (see
[`docs/migration_log.md`](docs/migration_log.md)). Stack: Python 3.10,
scikit-learn, lifelines, statsmodels, pandas, numpy, matplotlib, neuroCombat.

## Disclaimer

This repository is for research and educational purposes only. Patient-level
imaging and clinical data are not publicly distributed due to privacy and
institutional restrictions.

