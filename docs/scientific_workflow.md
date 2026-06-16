# Scientific Workflow

This document formalizes the analysis logic (previously captured informally in
`Notes.ini`). The methodology, hyperparameters, seeds, and statistical
procedures are unchanged by the refactor — only the engineering around them.

## Pipeline data flow

```
raw data (Clinical, Imaging_Features, Annotation_Boxes)
   │  preprocessing/merge_data.py
   ▼
merged_dataset.csv
   │  preprocessing/prepare_radiomics.py  (QC → median impute → z-score)
   ▼
z-scored radiomic feature matrix
   │  clustering/pca.py  (randomized PCA, seed=42)
   ▼
PCA latent space
   │  clustering/kmeans.py  (k-sweep 2..6, final k=2, n_init=20)
   ▼
stable 2-cluster phenotype  ──►  clustering/stability.py  (bootstrap ARI)
   │
   ├─► statistics/biology_analysis.py   enhancement-kinetic dominant features
   │                                    (Welch t-test, Cohen's d, BH-FDR)
   ├─► statistics/manufacturer_analysis.py   cluster × manufacturer χ²
   ├─► statistics/survival.py           HR+/HER2+ recurrence-free survival
   │                                    (KM, log-rank, Cox ± manufacturer)
   └─► harmonization/                   residualization, ComBat, within-vendor,
                                        cross-condition stability
```

## The logic chain of the study

| Phase | Step | Module |
|-------|------|--------|
| 1. Discovery | Unsupervised clustering finds a stable phenotype | `clustering/` |
| 2. Interpretation | Phenotype appears vascular (SER-map features) and prognostic | `statistics/biology_analysis.py`, `statistics/survival.py` |
| 3. Confounding | Phenotype overlaps almost perfectly with manufacturer | `statistics/manufacturer_analysis.py` |
| 4. Re-evaluation | Harmonization collapses the partition; biology becomes uncertain | `harmonization/` |

## Interpretation table

| Result | Naive interpretation | After confounding analysis |
|--------|---------------------|----------------------------|
| Stable clusters (bootstrap ARI ≈ 0.89) | looks real | stability is necessary, not sufficient |
| Vascular enhancement features dominate | looks biological | features track acquisition |
| Worse HR+/HER2+ outcome | looks prognostic | not robust to manufacturer adjustment |
| Cluster × manufacturer χ² p ≈ 1e-104 | — | acquisition confounding |

## Core methodological conclusion

> **Cluster stability and cluster biological identity are independent
> properties.** A radiomic phenotype can be highly bootstrap-stable while
> being acquisition-driven rather than biologically meaningful. After
> manufacturer harmonization, bootstrap stability remains high (~0.90) yet the
> partition identity collapses (ARI vs raw ≈ 0.06).
