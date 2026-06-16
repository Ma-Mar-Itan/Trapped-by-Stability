"""Stage 2 -- clustering: PCA, KMeans phenotype discovery, bootstrap stability.

Requires the preprocessing stage to have run first.

Usage:
    python scripts/run_clustering.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from radiomics_stability.clustering import kmeans, pca, stability


def main() -> None:
    pca.main()
    kmeans.main()
    stability.main()


if __name__ == "__main__":
    main()
