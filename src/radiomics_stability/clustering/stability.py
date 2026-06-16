"""Bootstrap stability of the discovered phenotype.

Migrated from ``4-Cluster stability.py``. Repeatedly subsamples 80% of patients,
re-clusters, and measures the Adjusted Rand Index (ARI) against the baseline
labels on the same subset. The :func:`bootstrap_ari` routine is reused by the
harmonization validation stage to compare stability across conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from radiomics_stability import paths
from radiomics_stability.clustering.pca import load_feature_matrix, run_pca
from radiomics_stability.config import (
    KMEANS_N_INIT,
    N_BOOTSTRAPS,
    N_CLUSTERS,
    N_PCS_CLUSTERING,
    RANDOM_SEED,
    SUBSAMPLE_FRACTION,
)
from radiomics_stability.io_utils import save_csv, save_figure
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def bootstrap_ari(
    x_matrix: np.ndarray,
    base_labels: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    frac: float = SUBSAMPLE_FRACTION,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Return per-iteration ARI between re-clustered subsamples and ``base_labels``.

    For each iteration ``b``, a random ``frac`` subsample (without replacement)
    is re-clustered with ``random_state = seed + b`` and compared to the
    baseline labels restricted to the same subset.
    """
    rng = np.random.default_rng(seed)
    n = x_matrix.shape[0]
    scores = []
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        sub_labels = KMeans(
            n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=seed + b
        ).fit_predict(x_matrix[idx])
        scores.append(adjusted_rand_score(base_labels[idx], sub_labels))
    return np.array(scores)


def _plot_bootstrap(scores: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = np.arange(1, len(scores) + 1)
    ax.plot(iterations, scores, marker="o")
    ax.axhline(scores.mean(), linestyle="--")
    ax.set_xlabel("Bootstrap Iteration")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Bootstrap Stability of k=2 Phenotype")
    ax.grid(True)
    return fig


def main(save_figures: bool = True) -> pd.DataFrame:
    """Compute and persist bootstrap ARI scores for the baseline phenotype."""
    paths.ensure_dirs()
    _, x = load_feature_matrix()
    scores_pca, _ = run_pca(x)
    x_cluster = scores_pca[:, :N_PCS_CLUSTERING]

    baseline_labels = KMeans(
        n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED
    ).fit_predict(x_cluster)

    ari_scores = bootstrap_ari(x_cluster, baseline_labels)
    logger.info("Bootstrap ARI: mean=%.4f sd=%.4f", ari_scores.mean(), ari_scores.std())

    ari_df = pd.DataFrame(
        {"bootstrap": np.arange(1, len(ari_scores) + 1), "ARI": ari_scores}
    )
    save_csv(ari_df, paths.BOOTSTRAP_ARI_CSV)

    if save_figures:
        save_figure(_plot_bootstrap(ari_scores), paths.FIG_BOOTSTRAP)

    return ari_df


if __name__ == "__main__":
    main()
