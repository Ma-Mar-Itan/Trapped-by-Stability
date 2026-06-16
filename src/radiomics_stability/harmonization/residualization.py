"""Manufacturer residualization (linear confound removal).

Migrated from ``10- adjustment for mani.py``. Regresses each feature on the
manufacturer design matrix, keeps the residuals, re-standardizes, then re-runs
PCA + KMeans. The ARI of the residualized partition against the raw phenotype
quantifies how much of the phenotype was manufacturer-driven.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from radiomics_stability import paths
from radiomics_stability.clustering.pca import run_pca
from radiomics_stability.config import (
    CLUSTER_COL,
    KMEANS_N_INIT,
    N_CLUSTERS,
    N_PCS_CLUSTERING,
    PATIENT_ID_COL,
    RANDOM_SEED,
)
from radiomics_stability.harmonization.frame import AnalysisFrame, build_harmonization_frame
from radiomics_stability.io_utils import save_csv, save_figure
from radiomics_stability.logging_utils import get_logger
from radiomics_stability.visualization.plots import scatter_by_group

logger = get_logger(__name__)


@dataclass
class ResidualizationResult:
    labels: np.ndarray
    x_resid_scaled: pd.DataFrame
    x_resid_pca: np.ndarray
    ari_vs_raw: float


def residualize_features(x: pd.DataFrame, manufacturer: pd.Series) -> pd.DataFrame:
    """Remove the manufacturer-associated linear component from each feature."""
    design = pd.get_dummies(manufacturer, drop_first=True).astype(float)
    lr = LinearRegression()
    lr.fit(design.values, x.values)
    residuals = x.values - lr.predict(design.values)
    return pd.DataFrame(residuals, columns=x.columns, index=x.index)


def run_residualization(frame: AnalysisFrame) -> ResidualizationResult:
    """Residualize, re-standardize, re-cluster, and compare to the raw phenotype."""
    x_resid = residualize_features(frame.x, frame.manufacturer)

    x_resid_scaled = pd.DataFrame(
        StandardScaler().fit_transform(x_resid),
        columns=frame.feature_cols,
        index=frame.df.index,
    )
    x_resid_pca, _ = run_pca(x_resid_scaled)
    labels = KMeans(
        n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED
    ).fit_predict(x_resid_pca[:, :N_PCS_CLUSTERING])

    ari = adjusted_rand_score(frame.raw_labels, labels)
    logger.info("ARI between raw clusters and residualized clusters: %.4f", ari)
    return ResidualizationResult(labels, x_resid_scaled, x_resid_pca, ari)


def _plot_comparison(frame: AnalysisFrame, result: ResidualizationResult):
    import matplotlib.pyplot as plt

    raw_pca, _ = run_pca(frame.x)
    pc1, pc2 = raw_pca[:, 0], raw_pca[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

    scatter_by_group(axes[0], pc1, pc2, frame.raw_labels, sort_groups=True, label_fmt="Cluster {}")
    axes[0].set_title("A. Raw Radiomic Clusters")
    axes[0].set_ylabel("PC2")

    scatter_by_group(axes[1], pc1, pc2, frame.manufacturer.values)
    axes[1].set_title("B. Manufacturer")

    scatter_by_group(axes[2], pc1, pc2, result.labels, sort_groups=True, label_fmt="Cluster {}")
    axes[2].set_title(
        f"C. Clusters After Manufacturer Residualization\nARI vs Raw = {result.ari_vs_raw:.3f}"
    )

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.legend()
    fig.tight_layout()
    return fig


def persist(frame: AnalysisFrame, result: ResidualizationResult, save_figures: bool = True) -> None:
    """Write residualized cluster labels and the re-standardized feature matrix."""
    save_csv(
        pd.DataFrame(
            {
                PATIENT_ID_COL: frame.df[PATIENT_ID_COL].values,
                "Raw_Cluster": frame.raw_labels,
                "Residualized_Cluster": result.labels,
                "Manufacturer": frame.manufacturer.values,
            }
        ),
        paths.CLUSTER_LABELS_RESID_CSV,
    )

    x_out = result.x_resid_scaled.copy()
    x_out.insert(0, PATIENT_ID_COL, frame.df[PATIENT_ID_COL].values)
    save_csv(x_out, paths.X_RESID_CSV)

    if save_figures:
        save_figure(_plot_comparison(frame, result), paths.FIG_RESID_COMPARISON)


def main(save_figures: bool = True) -> ResidualizationResult:
    paths.ensure_dirs()
    frame = build_harmonization_frame()
    result = run_residualization(frame)
    persist(frame, result, save_figures=save_figures)
    return result


if __name__ == "__main__":
    main()
