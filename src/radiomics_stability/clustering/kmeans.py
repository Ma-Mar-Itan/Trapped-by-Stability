"""KMeans phenotype discovery on the PCA latent space.

Migrated from ``3-Kmeans.py``. Sweeps k = 2..6 computing internal validity
metrics, then fits the final 2-cluster solution that defines the discovered
phenotype. The k=2 labels are the baseline used by every downstream analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from radiomics_stability import paths
from radiomics_stability.clustering.pca import load_feature_matrix, run_pca
from radiomics_stability.config import (
    K_RANGE,
    KMEANS_N_INIT,
    N_CLUSTERS,
    N_PCS_CLUSTERING,
    PATIENT_ID_COL,
    RANDOM_SEED,
)
from radiomics_stability.config import CLUSTER_COL
from radiomics_stability.io_utils import save_csv, save_figure
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def fit_kmeans(x_cluster: np.ndarray, k: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Fit KMeans with the configured ``n_init`` and return cluster labels."""
    model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed)
    return model.fit_predict(x_cluster)


def evaluate_k_range(x_cluster: np.ndarray) -> pd.DataFrame:
    """Compute silhouette / Calinski-Harabasz / Davies-Bouldin over ``K_RANGE``."""
    results = []
    for k in K_RANGE:
        labels = fit_kmeans(x_cluster, k)
        results.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_cluster, labels),
                "calinski_harabasz": calinski_harabasz_score(x_cluster, labels),
                "davies_bouldin": davies_bouldin_score(x_cluster, labels),
            }
        )
    return pd.DataFrame(results)


def _plot_metrics(results_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(
        axes,
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"],
    ):
        ax.plot(results_df["k"], results_df[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("k")
    fig.tight_layout()
    return fig


def main(save_figures: bool = True) -> pd.DataFrame:
    """Run model selection, fit final k=2, and persist labels + metrics."""
    paths.ensure_dirs()
    patient_ids, x = load_feature_matrix()
    scores, _ = run_pca(x)
    x_cluster = scores[:, :N_PCS_CLUSTERING]
    logger.info("Clustering matrix shape: %s", x_cluster.shape)

    metrics = evaluate_k_range(x_cluster)
    logger.info("Cluster validity results:\n%s", metrics.to_string(index=False))
    save_csv(metrics, paths.KMEANS_METRICS_CSV)

    final_labels = fit_kmeans(x_cluster, N_CLUSTERS)
    cluster_df = pd.DataFrame(
        {PATIENT_ID_COL: patient_ids, CLUSTER_COL: final_labels}
    )
    save_csv(cluster_df, paths.CLUSTER_LABELS_K2_CSV)

    if save_figures:
        save_figure(_plot_metrics(metrics), paths.FIG_KMEANS_METRICS)

    return cluster_df


if __name__ == "__main__":
    main()
