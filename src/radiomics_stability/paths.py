"""Centralized, machine-independent filesystem paths.

The project root is derived from this file's location, so the pipeline runs
unchanged on any machine and from any working directory. Both the root and the
data directory can be overridden with environment variables, which is handy for
CI or for pointing at a data drive:

* ``RADSTAB_PROJECT_ROOT`` -- override the inferred project root
* ``RADSTAB_DATA_DIR``     -- override the data directory specifically
"""

from __future__ import annotations

import os
from pathlib import Path

# src/radiomics_stability/paths.py -> parents[2] == project root
_INFERRED_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT: Path = Path(
    os.environ.get("RADSTAB_PROJECT_ROOT", _INFERRED_ROOT)
).resolve()

DATA_DIR: Path = Path(
    os.environ.get("RADSTAB_DATA_DIR", PROJECT_ROOT / "data")
).resolve()

# Stage-oriented data subdirectories ---------------------------------------
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = DATA_DIR / "results"

# --- Raw inputs (tracked; never written by the pipeline) -------------------
CLINICAL_CSV: Path = RAW_DIR / "Clinical.csv"
IMAGING_FEATURES_CSV: Path = RAW_DIR / "Imaging_Features.csv"
ANNOTATION_BOXES_CSV: Path = RAW_DIR / "Annotation_Boxes.csv"

# --- Interim ---------------------------------------------------------------
MERGED_DATASET_CSV: Path = INTERIM_DIR / "merged_dataset.csv"

# --- Processed (analysis-ready matrices) -----------------------------------
X_ZSCORED_CSV: Path = PROCESSED_DIR / "X_radiomics_zscored.csv"
X_ZSCORED_WITH_IDS_CSV: Path = PROCESSED_DIR / "X_radiomics_zscored_with_ids.csv"
PATIENT_IDS_CSV: Path = PROCESSED_DIR / "patient_ids.csv"
FEATURE_LIST_CSV: Path = PROCESSED_DIR / "radiomic_feature_list.csv"
PCA_SCORES_CSV: Path = PROCESSED_DIR / "PCA_scores.csv"
X_RESID_CSV: Path = PROCESSED_DIR / "X_radiomics_residualized_manufacturer_zscored.csv"
X_COMBAT_CSV: Path = PROCESSED_DIR / "X_radiomics_combat_zscored.csv"

# --- Results (tables + figures) --------------------------------------------
CLUSTER_LABELS_K2_CSV: Path = RESULTS_DIR / "cluster_labels_k2.csv"
KMEANS_METRICS_CSV: Path = RESULTS_DIR / "kmeans_validity_metrics.csv"
BOOTSTRAP_ARI_CSV: Path = RESULTS_DIR / "bootstrap_ari_results.csv"
TOP_FEATURES_CSV: Path = RESULTS_DIR / "top_cluster_features.csv"

CLUSTER_ASSOCIATION_CSV: Path = RESULTS_DIR / "cluster_association_results.csv"
CLUSTER_BY_MANUF_COUNTS_CSV: Path = RESULTS_DIR / "cluster_by_manufacturer_counts.csv"
CLUSTER_BY_MANUF_WITHIN_CLUSTER_CSV: Path = (
    RESULTS_DIR / "cluster_by_manufacturer_percentages_within_cluster.csv"
)
CLUSTER_BY_MANUF_WITHIN_MANUF_CSV: Path = (
    RESULTS_DIR / "cluster_by_manufacturer_percentages_within_manufacturer.csv"
)
CLUSTER_MANUF_PRESENTATION_CSV: Path = (
    RESULTS_DIR / "cluster_manufacturer_presentation_table.csv"
)

SURVIVAL_SUMMARY_CSV: Path = RESULTS_DIR / "survival_outcome_summary.csv"
COX_CLUSTER_ONLY_CSV: Path = RESULTS_DIR / "cox_cluster_only.csv"
COX_MANUFACTURER_ADJUSTED_CSV: Path = RESULTS_DIR / "cox_manufacturer_adjusted.csv"

CLUSTER_LABELS_RESID_CSV: Path = RESULTS_DIR / "cluster_labels_residualized_manufacturer.csv"
CLUSTER_LABELS_COMBAT_CSV: Path = RESULTS_DIR / "cluster_labels_combat.csv"
WITHIN_MANUF_RESULTS_CSV: Path = RESULTS_DIR / "within_manufacturer_clustering_results.csv"
ASSOCIATION_AFTER_HARM_CSV: Path = RESULTS_DIR / "association_after_harmonization.csv"
STABILITY_SUMMARY_CSV: Path = RESULTS_DIR / "stability_summary_raw_resid_combat.csv"

# --- Figures ---------------------------------------------------------------
FIG_SCREE: Path = RESULTS_DIR / "pca_scree_plot.png"
FIG_PCA_SCATTER: Path = RESULTS_DIR / "pca_pc1_pc2_scatter.png"
FIG_KMEANS_METRICS: Path = RESULTS_DIR / "kmeans_validity_metrics.png"
FIG_BOOTSTRAP: Path = RESULTS_DIR / "bootstrap_stability_k2.png"
FIG_CLUSTER_MANUF_OVERLAP: Path = RESULTS_DIR / "cluster_manufacturer_overlap_pca.png"
FIG_KM_CURVES: Path = RESULTS_DIR / "km_rfs_hrher2_by_cluster.png"
FIG_RESID_COMPARISON: Path = RESULTS_DIR / "raw_manufacturer_residualized_cluster_comparison.png"
FIG_STABILITY_CONDITIONS: Path = RESULTS_DIR / "stability_raw_resid_combat.png"


def within_manufacturer_labels_csv(manufacturer: str) -> Path:
    """Path of the per-manufacturer within-vendor cluster-label file."""
    return RESULTS_DIR / f"cluster_labels_within_{manufacturer}.csv"


def ensure_dirs() -> None:
    """Create the generated-data directories if they do not yet exist."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
