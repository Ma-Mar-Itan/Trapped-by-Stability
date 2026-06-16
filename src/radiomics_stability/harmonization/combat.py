"""ComBat harmonization of manufacturer batch effects.

Migrated from the first half of ``11- ComBat harmonization.py``. Applies
neuroCombat with manufacturer as the batch variable, then re-standardizes,
re-runs PCA + KMeans, and compares to the raw phenotype. The harmonized feature
matrix is persisted so the validation stage can recompute the ComBat PCA
without re-running neuroCombat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from radiomics_stability import paths
from radiomics_stability.clustering.pca import run_pca
from radiomics_stability.config import (
    KMEANS_N_INIT,
    MANUFACTURER_COL,
    N_CLUSTERS,
    N_PCS_CLUSTERING,
    PATIENT_ID_COL,
    RANDOM_SEED,
)
from radiomics_stability.harmonization.frame import AnalysisFrame, build_harmonization_frame
from radiomics_stability.io_utils import save_csv
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CombatResult:
    labels: np.ndarray
    x_combat_scaled: pd.DataFrame
    x_combat_pca: np.ndarray
    ari_vs_raw: float


def apply_combat(x: pd.DataFrame, manufacturer: pd.Series) -> pd.DataFrame:
    """Run neuroCombat with manufacturer as the batch column.

    Imported lazily so the package is importable without neuroCombat installed.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "neuroCombat is required for ComBat harmonization. "
            "Install it with `pip install neuroCombat`."
        ) from exc

    covars = pd.DataFrame({MANUFACTURER_COL: manufacturer.values})
    result = neuroCombat(dat=x.T, covars=covars, batch_col=MANUFACTURER_COL)
    return pd.DataFrame(result["data"].T, columns=x.columns, index=x.index)


def run_combat(frame: AnalysisFrame) -> CombatResult:
    """Harmonize, re-standardize, re-cluster, and compare to the raw phenotype."""
    x_combat = apply_combat(frame.x, frame.manufacturer)

    x_combat_scaled = pd.DataFrame(
        StandardScaler().fit_transform(x_combat),
        columns=frame.feature_cols,
        index=frame.df.index,
    )
    x_combat_pca, _ = run_pca(x_combat_scaled)
    labels = KMeans(
        n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED
    ).fit_predict(x_combat_pca[:, :N_PCS_CLUSTERING])

    ari = adjusted_rand_score(frame.raw_labels, labels)
    logger.info("ComBat ARI vs raw: %.4f", ari)
    return CombatResult(labels, x_combat_scaled, x_combat_pca, ari)


def persist(frame: AnalysisFrame, result: CombatResult) -> None:
    """Write ComBat cluster labels and the harmonized feature matrix."""
    save_csv(
        pd.DataFrame(
            {
                PATIENT_ID_COL: frame.df[PATIENT_ID_COL].values,
                "Raw_Cluster": frame.raw_labels,
                "ComBat_Cluster": result.labels,
                "Manufacturer": frame.manufacturer.values,
            }
        ),
        paths.CLUSTER_LABELS_COMBAT_CSV,
    )

    x_out = result.x_combat_scaled.copy()
    x_out.insert(0, PATIENT_ID_COL, frame.df[PATIENT_ID_COL].values)
    save_csv(x_out, paths.X_COMBAT_CSV)


def main() -> CombatResult:
    paths.ensure_dirs()
    frame = build_harmonization_frame()
    result = run_combat(frame)
    persist(frame, result)
    return result


if __name__ == "__main__":
    main()
