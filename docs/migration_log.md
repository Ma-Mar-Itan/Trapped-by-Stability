# Migration Log

Refactor of the original Spyder-style script collection into a reproducible,
installable Python package. **No scientific methodology, hyperparameter, seed,
statistical test, or numerical result was changed.** Every regenerated artifact
was diffed against the original committed outputs (see *Verification* below).

## Script → module mapping

| Old file | New location | Purpose | Notable changes |
|----------|--------------|---------|-----------------|
| `Join all data sets.py` | `src/radiomics_stability/preprocessing/merge_data.py` | Merge raw tables | Paths centralized; returns/saves explicitly |
| `1-Prepare Radiomics Features.py` | `preprocessing/prepare_radiomics.py` | QC + impute + z-score | Removed `!pip install`; QC thresholds → `config` |
| `2-PCA.py` | `clustering/pca.py` | Randomized PCA | `X_pca` no longer a global; recomputed deterministically |
| `3-Kmeans.py` | `clustering/kmeans.py` | k-sweep + final k=2 | Metrics table now saved (`kmeans_validity_metrics.csv`) |
| `4-Cluster stability.py` | `clustering/stability.py` | Bootstrap ARI | `bootstrap_ari()` extracted and reused by validation |
| `5-biology discovery.py` | `statistics/biology_analysis.py` | Welch t-test / Cohen's d / FDR | Per-feature test extracted to a function |
| `Manifcature correlation.py` + `7-manifacture.py` | `statistics/manufacturer_analysis.py` | χ² association + crosstabs + overlap figure | **Merged** (overlapping work); duplicate crosstab code removed |
| `6- death associated.py` + `9- cox model.py` | `statistics/survival.py` | KM, log-rank, Cox (± manufacturer) | **Merged**; duplicated RFS-time construction → `build_rfs_frame()` |
| `10- adjustment for mani.py` | `harmonization/residualization.py` | Linear residualization | Removed `!pip install`; uses shared analysis frame |
| `11- ComBat harmonization.py` | `harmonization/combat.py` + `harmonization/validation.py` | ComBat + within-vendor + association + stability | **Split** into harmonization vs validation; shared frame |
| `Notes.ini` | `docs/scientific_workflow.md` | Study logic chain | Formalized as Markdown |

## New supporting modules (no original equivalent)

| File | Role |
|------|------|
| `config.py` | Single source of truth for seeds, hyperparameters, thresholds, label maps, column names |
| `paths.py` | Project-root-relative paths with env-var overrides; replaces hard-coded `C:\Users\malek\...` paths |
| `logging_utils.py` | One logging configuration; replaces scattered `print` calls |
| `io_utils.py` | Typed `load_csv` / `save_csv` / `save_figure` helpers |
| `transforms.py` | Shared `clean_manufacturer`, `chi2_association`, `crosstab_percentages` |
| `harmonization/frame.py` | Builds the aligned analysis frame once (replaces cross-script in-memory globals) |
| `visualization/plots.py` | Shared `scatter_by_group` primitive |
| `scripts/*.py` | Thin CLI entry points per stage + `run_full_pipeline.py` |

## Technical-debt fixes

- **Hard-coded absolute paths** removed everywhere → `paths.py` (machine-independent, env-overridable).
- **Hidden in-memory coupling** eliminated: every module imports its own inputs, loads from disk, and returns explicit outputs. The original chains (`2→3→4→7→10→11`, `6→9`) that relied on leftover Spyder variables (`X_pca`, `df`, `raw_labels`, …) now reconstruct state deterministically via `clustering.pca.run_pca` and `harmonization.frame.build_harmonization_frame`.
- **Jupyter magics** (`!pip install ...`) removed; dependencies declared in `requirements.txt` / `pyproject.toml`.
- **Duplicate logic** consolidated: RFS-time construction, manufacturer cleaning, crosstab/percentage generation, scatter-by-group plotting.
- **Dependency mismatch** fixed: `neuroHarmonize` → `neuroCombat` (the package actually imported).
- **Committed `.venv`** untracked (`git rm --cached`) and added to `.gitignore`.
- **Naming** standardized to `snake_case` modules (e.g. `6- death associated.py` → `survival.py`).
- **Data layout** reorganized into `raw/ → interim/ → processed/ → results/`. The duplicate `data/Radiomic Features Data/` folder and scattered root-level outputs were removed; their content is regenerated identically under `data/processed` and `data/results`.

## Data directory reorganization

| Old | New |
|-----|-----|
| `data/raw/*` | `data/raw/*` (unchanged — inputs) |
| `data/merged_dataset.csv` | `data/interim/merged_dataset.csv` |
| `data/X_radiomics_zscored*.csv`, `patient_ids.csv`, `radiomic_feature_list.csv`, `PCA_scores.csv` | `data/processed/` |
| `data/Radiomic Features Data/*` (duplicate) | removed |
| all `cluster_*`, `*association*`, `stability_*`, `top_cluster_features.csv`, `*.png` | `data/results/` |

## Verification

Each stage was regenerated from raw data and compared against the original
committed outputs. All numeric artifacts matched to floating-point exactness
(`max_abs_diff = 0.0`); cluster-label files matched with `ARI = 1.0`
(label-permutation invariant). Verified artifacts include the z-scored matrix,
`cluster_labels_k2`, `bootstrap_ari_results`, `top_cluster_features`,
`cluster_association_results`, `cluster_by_manufacturer_counts`,
`within_manufacturer_clustering_results`, `association_after_harmonization`,
`stability_summary_raw_resid_combat`, the residualized feature matrix, and the
residualized / ComBat cluster labels.

Environment used for verification: Python 3.10, scikit-learn 1.7.2,
neuroCombat. Results are sensitive to library versions (KMeans, randomized
PCA), so `requirements.txt` should be respected for exact reproduction.
