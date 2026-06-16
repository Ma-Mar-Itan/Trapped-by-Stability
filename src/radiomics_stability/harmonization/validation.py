"""Validation synthesis across raw and harmonized conditions.

Migrated from the second half of ``11- ComBat harmonization.py``. Brings
together the three analyses that close the argument:

1. Within-manufacturer clustering (does the phenotype survive inside a vendor?).
2. Association of each cluster definition with manufacturer / subtype / recurrence.
3. Bootstrap stability across raw, residualized, and ComBat conditions.

The headline result: stability stays high in every condition while the
partition identity (ARI vs raw) collapses after harmonization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from radiomics_stability import paths
from radiomics_stability.clustering.pca import run_pca
from radiomics_stability.clustering.stability import bootstrap_ari
from radiomics_stability.config import (
    CLUSTER_COL,
    EVENT_COL,
    KMEANS_N_INIT,
    MANUFACTURER_CLEAN_COL,
    N_CLUSTERS,
    N_PCS_CLUSTERING,
    PATIENT_ID_COL,
    PCA_SVD_SOLVER,
    RANDOM_SEED,
    SUBTYPE_COL,
)
from radiomics_stability.harmonization import combat as combat_mod
from radiomics_stability.harmonization import residualization as resid_mod
from radiomics_stability.harmonization.combat import run_combat
from radiomics_stability.harmonization.frame import AnalysisFrame, build_harmonization_frame
from radiomics_stability.harmonization.residualization import run_residualization
from radiomics_stability.io_utils import load_csv, save_csv, save_figure
from radiomics_stability.logging_utils import get_logger
from radiomics_stability.transforms import chi2_association

logger = get_logger(__name__)


def within_manufacturer_clustering(frame: AnalysisFrame) -> pd.DataFrame:
    """Re-cluster within each manufacturer and compare to the raw phenotype."""
    results = []
    for manufacturer in frame.manufacturer.unique():
        idx = (frame.manufacturer == manufacturer).values
        x_sub = frame.x.loc[idx].copy()
        raw_sub = frame.raw_labels[idx]

        x_sub_scaled = pd.DataFrame(
            StandardScaler().fit_transform(x_sub),
            columns=frame.feature_cols,
            index=x_sub.index,
        )
        x_sub_pca = PCA(svd_solver=PCA_SVD_SOLVER, random_state=RANDOM_SEED).fit_transform(
            x_sub_scaled
        )
        sub_labels = KMeans(
            n_clusters=N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED
        ).fit_predict(x_sub_pca[:, :N_PCS_CLUSTERING])

        results.append(
            {
                "Manufacturer": manufacturer,
                "n": int(idx.sum()),
                "ARI_vs_raw": adjusted_rand_score(raw_sub, sub_labels),
                "Silhouette": silhouette_score(x_sub_pca[:, :N_PCS_CLUSTERING], sub_labels),
            }
        )
        save_csv(
            pd.DataFrame(
                {
                    PATIENT_ID_COL: frame.df.loc[idx, PATIENT_ID_COL],
                    "Raw_Cluster": raw_sub,
                    "Within_Manufacturer_Cluster": sub_labels,
                    "Manufacturer": manufacturer,
                }
            ),
            paths.within_manufacturer_labels_csv(manufacturer),
        )

    results_df = pd.DataFrame(results)
    save_csv(results_df, paths.WITHIN_MANUF_RESULTS_CSV)
    logger.info("Within-manufacturer clustering:\n%s", results_df.to_string(index=False))
    return results_df


def association_after_harmonization(
    frame: AnalysisFrame, resid_labels: np.ndarray, combat_labels: np.ndarray
) -> pd.DataFrame:
    """Chi-square association of each cluster definition with key variables."""
    merged = load_csv(paths.MERGED_DATASET_CSV)
    harm_df = frame.df.copy()
    harm_df["Residualized_Cluster"] = resid_labels
    harm_df["ComBat_Cluster"] = combat_labels
    harm_df = harm_df.merge(
        merged[[PATIENT_ID_COL, SUBTYPE_COL, EVENT_COL]], on=PATIENT_ID_COL, how="left"
    )

    tests = []
    for cluster_col in [CLUSTER_COL, "Residualized_Cluster", "ComBat_Cluster"]:
        for variable_col in [MANUFACTURER_CLEAN_COL, SUBTYPE_COL, EVENT_COL]:
            _table, p, _chi2, _dof = chi2_association(harm_df, cluster_col, variable_col)
            tests.append({"Cluster_Type": cluster_col, "Variable": variable_col, "p_value": p})
            logger.info("%s vs %s: p=%.4g", cluster_col, variable_col, p)

    out = pd.DataFrame(tests)
    save_csv(out, paths.ASSOCIATION_AFTER_HARM_CSV)
    return out


def stability_across_conditions(
    raw_pca: np.ndarray,
    raw_labels: np.ndarray,
    resid_pca: np.ndarray,
    resid_labels: np.ndarray,
    combat_pca: np.ndarray,
    combat_labels: np.ndarray,
    save_figures: bool = True,
) -> pd.DataFrame:
    """Bootstrap stability for raw / residualized / ComBat conditions."""
    raw_boot = bootstrap_ari(raw_pca[:, :N_PCS_CLUSTERING], raw_labels)
    resid_boot = bootstrap_ari(resid_pca[:, :N_PCS_CLUSTERING], resid_labels)
    combat_boot = bootstrap_ari(combat_pca[:, :N_PCS_CLUSTERING], combat_labels)

    summary = pd.DataFrame(
        {
            "Condition": ["Raw", "Residualized", "ComBat"],
            "Mean_ARI": [raw_boot.mean(), resid_boot.mean(), combat_boot.mean()],
            "SD_ARI": [raw_boot.std(), resid_boot.std(), combat_boot.std()],
            "ARI_vs_raw": [
                1.0,
                adjusted_rand_score(raw_labels, resid_labels),
                adjusted_rand_score(raw_labels, combat_labels),
            ],
        }
    )
    save_csv(summary, paths.STABILITY_SUMMARY_CSV)
    logger.info("Stability summary:\n%s", summary.to_string(index=False))

    if save_figures:
        save_figure(_plot_stability(summary), paths.FIG_STABILITY_CONDITIONS)
    return summary


def _plot_stability(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        summary["Condition"], summary["Mean_ARI"], yerr=summary["SD_ARI"], capsize=5
    )
    ax.set_ylabel("Bootstrap ARI")
    ax.set_title("Cluster Stability Across Raw and Harmonized Conditions")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main(save_figures: bool = True) -> pd.DataFrame:
    """Run the full harmonization + validation synthesis."""
    paths.ensure_dirs()
    frame = build_harmonization_frame()

    resid = run_residualization(frame)
    resid_mod.persist(frame, resid, save_figures=save_figures)
    combat = run_combat(frame)
    combat_mod.persist(frame, combat)
    raw_pca, _ = run_pca(frame.x)

    within_manufacturer_clustering(frame)
    association_after_harmonization(frame, resid.labels, combat.labels)
    summary = stability_across_conditions(
        raw_pca, frame.raw_labels,
        resid.x_resid_pca, resid.labels,
        combat.x_combat_pca, combat.labels,
        save_figures=save_figures,
    )
    return summary


if __name__ == "__main__":
    main()
