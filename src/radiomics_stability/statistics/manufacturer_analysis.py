"""Manufacturer-confounding analysis of the discovered phenotype.

Consolidates the two original scripts that performed overlapping work:

* ``Manifcature correlation.py`` -- chi-square of cluster vs manufacturer,
  molecular subtype, and recurrence.
* ``7-manifacture.py``           -- cluster x manufacturer crosstabs (counts and
  percentages) plus the PCA-overlap figure.

The duplicated crosstab/percentage logic is unified here via
:func:`radiomics_stability.transforms`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radiomics_stability import paths
from radiomics_stability.clustering.pca import load_feature_matrix, run_pca
from radiomics_stability.config import (
    CLUSTER_COL,
    EVENT_COL,
    MANUFACTURER_CLEAN_COL,
    MANUFACTURER_COL,
    PATIENT_ID_COL,
    SUBTYPE_COL,
)
from radiomics_stability.io_utils import load_csv, save_csv, save_figure
from radiomics_stability.logging_utils import get_logger
from radiomics_stability.transforms import (
    chi2_association,
    clean_manufacturer,
    crosstab_percentages,
)
from radiomics_stability.visualization.plots import scatter_by_group

logger = get_logger(__name__)


def run_associations(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Chi-square of cluster vs manufacturer, molecular subtype, recurrence."""
    rows = []
    for label, col in [
        ("Manufacturer", MANUFACTURER_COL),
        ("Molecular Subtype", SUBTYPE_COL),
        ("Recurrence", EVENT_COL),
    ]:
        _table, p, _chi2, _dof = chi2_association(df_clustered, CLUSTER_COL, col)
        logger.info("Cluster vs %s: p=%.3e", label, p)
        rows.append({"Variable": label, "P-value": p})
    return pd.DataFrame(rows)


def manufacturer_crosstabs(df_clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build counts, within-cluster %, within-manufacturer %, presentation table."""
    table = pd.crosstab(df_clean[CLUSTER_COL], df_clean[MANUFACTURER_CLEAN_COL])
    row_pct, col_pct = crosstab_percentages(table)
    presentation = row_pct.round(1).astype(str) + "%"
    return {
        "counts": table,
        "within_cluster": row_pct,
        "within_manufacturer": col_pct,
        "presentation": presentation,
    }


def _plot_overlap(pc1: np.ndarray, pc2: np.ndarray, clusters, manufacturers):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    scatter_by_group(axes[0], pc1, pc2, clusters, sort_groups=True, label_fmt="Cluster {}")
    axes[0].set_title("A. Radiomic Clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(title="Cluster")

    scatter_by_group(axes[1], pc1, pc2, manufacturers)
    axes[1].set_title("B. MRI Manufacturer")
    axes[1].set_xlabel("PC1")
    axes[1].legend(title="Manufacturer")

    fig.suptitle("Radiomic Phenotype and Manufacturer Overlap in PCA Space", fontsize=16)
    fig.tight_layout()
    return fig


def main(save_figures: bool = True) -> pd.DataFrame:
    """Run association tests, crosstabs, and the PCA overlap figure."""
    paths.ensure_dirs()
    merged = load_csv(paths.MERGED_DATASET_CSV)
    clusters = load_csv(paths.CLUSTER_LABELS_K2_CSV)
    df = merged.merge(clusters, on=PATIENT_ID_COL, how="inner")

    # Association tests (use raw manufacturer codes; p-values are label-invariant).
    associations = run_associations(df)
    save_csv(associations, paths.CLUSTER_ASSOCIATION_CSV)

    # Crosstabs on cleaned manufacturer labels.
    df = clean_manufacturer(df, MANUFACTURER_COL)
    tabs = manufacturer_crosstabs(df)
    save_csv(tabs["counts"], paths.CLUSTER_BY_MANUF_COUNTS_CSV, index=True)
    save_csv(tabs["within_cluster"], paths.CLUSTER_BY_MANUF_WITHIN_CLUSTER_CSV, index=True)
    save_csv(tabs["within_manufacturer"], paths.CLUSTER_BY_MANUF_WITHIN_MANUF_CSV, index=True)
    save_csv(tabs["presentation"], paths.CLUSTER_MANUF_PRESENTATION_CSV, index=True)
    logger.info("Cluster x Manufacturer counts:\n%s", tabs["counts"].to_string())

    if save_figures:
        # Recompute PCA and align cluster/manufacturer to the feature-matrix order.
        patient_ids, x = load_feature_matrix()
        scores, _ = run_pca(x)
        order = pd.DataFrame({PATIENT_ID_COL: patient_ids.values})
        meta = order.merge(
            df[[PATIENT_ID_COL, CLUSTER_COL, MANUFACTURER_CLEAN_COL]],
            on=PATIENT_ID_COL,
            how="left",
        )
        fig = _plot_overlap(
            scores[:, 0], scores[:, 1], meta[CLUSTER_COL], meta[MANUFACTURER_CLEAN_COL]
        )
        save_figure(fig, paths.FIG_CLUSTER_MANUF_OVERLAP)

    return associations


if __name__ == "__main__":
    main()
