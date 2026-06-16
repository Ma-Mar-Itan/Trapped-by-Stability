"""Principal Component Analysis of the z-scored radiomic matrix.

Migrated from ``2-PCA.py``. The PCA is deterministic (randomized solver with a
fixed seed), so any stage that needs the principal-component scores recomputes
them via :func:`run_pca` instead of relying on an in-memory ``X_pca`` left over
from a previous script -- this removes the original hidden-state coupling while
producing identical scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from radiomics_stability import paths
from radiomics_stability.config import (
    N_PCS_SAVED,
    PATIENT_ID_COL,
    PCA_SVD_SOLVER,
    RANDOM_SEED,
)
from radiomics_stability.io_utils import load_csv, save_csv, save_figure
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def load_feature_matrix() -> tuple[pd.Series, pd.DataFrame]:
    """Load the processed z-scored matrix, returning ``(patient_ids, X)``."""
    df = load_csv(paths.X_ZSCORED_WITH_IDS_CSV)
    patient_ids = df[PATIENT_ID_COL]
    x = df.drop(columns=[PATIENT_ID_COL])
    return patient_ids, x


def run_pca(x: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, PCA]:
    """Fit randomized PCA and return ``(scores, fitted_model)``.

    Uses the configured solver and seed so the result is reproducible and
    matches the original ``2-PCA.py`` output exactly.
    """
    pca = PCA(svd_solver=PCA_SVD_SOLVER, random_state=RANDOM_SEED)
    scores = pca.fit_transform(x)
    return scores, pca


def _plot_scree(explained_variance: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 21), explained_variance[:20], marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Top 20 Principal Components")
    ax.grid(True)
    return fig


def _plot_pc_scatter(scores: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection: PC1 vs PC2")
    ax.grid(True)
    return fig


def main(save_figures: bool = True) -> np.ndarray:
    """Run PCA, persist the leading-PC scores and diagnostic figures."""
    paths.ensure_dirs()
    patient_ids, x = load_feature_matrix()
    scores, pca = run_pca(x)

    explained = pca.explained_variance_ratio_
    logger.info("Top 10 PCs explained variance:")
    for i in range(10):
        logger.info("  PC%d: %.4f", i + 1, explained[i])

    pca_df = pd.DataFrame(
        {PATIENT_ID_COL: patient_ids.values}
        | {f"PC{i + 1}": scores[:, i] for i in range(N_PCS_SAVED)}
    )
    save_csv(pca_df, paths.PCA_SCORES_CSV)

    if save_figures:
        save_figure(_plot_scree(explained), paths.FIG_SCREE)
        save_figure(_plot_pc_scatter(scores), paths.FIG_PCA_SCATTER)

    return scores


if __name__ == "__main__":
    main()
