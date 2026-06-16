"""Biological characterization of the discovered phenotype.

Migrated from ``5-biology discovery.py``. For every radiomic feature it runs a
Welch t-test between the two clusters, computes Cohen's d (pooled SD), applies
Benjamini-Hochberg FDR correction, and ranks features by absolute effect size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from radiomics_stability import paths
from radiomics_stability.config import (
    CLUSTER_COL,
    FDR_METHOD,
    N_TOP_FEATURES_DISPLAY,
    PATIENT_ID_COL,
)
from radiomics_stability.io_utils import load_csv, save_csv
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def _feature_test(x0: pd.Series, x1: pd.Series, feature: str) -> dict:
    """Welch t-test + Cohen's d for one feature between clusters 0 and 1."""
    x0, x1 = x0.dropna(), x1.dropna()
    mean0, mean1 = x0.mean(), x1.mean()
    sd0, sd1 = x0.std(ddof=1), x1.std(ddof=1)
    n0, n1 = len(x0), len(x1)

    t_stat, p_value = ttest_ind(x0, x1, equal_var=False, nan_policy="omit")

    pooled_sd = np.sqrt(((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / (n0 + n1 - 2))
    cohens_d = np.nan if pooled_sd == 0 else (mean1 - mean0) / pooled_sd

    return {
        "feature": feature,
        "mean_cluster0": mean0,
        "mean_cluster1": mean1,
        "difference_cluster1_minus_cluster0": mean1 - mean0,
        "cohens_d": cohens_d,
        "abs_cohens_d": abs(cohens_d),
        "p_value": p_value,
    }


def compare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run per-feature cluster comparison, FDR-correct, and sort by effect size."""
    cluster0 = df[df[CLUSTER_COL] == 0]
    cluster1 = df[df[CLUSTER_COL] == 1]
    logger.info("Cluster 0 n=%d  Cluster 1 n=%d", cluster0.shape[0], cluster1.shape[0])

    results = pd.DataFrame(
        [_feature_test(cluster0[f], cluster1[f], f) for f in feature_cols]
    )
    results["q_value"] = multipletests(results["p_value"], method=FDR_METHOD)[1]
    return results.sort_values(by="abs_cohens_d", ascending=False)


def main() -> pd.DataFrame:
    """Load features + clusters, run the comparison, and persist the ranking."""
    paths.ensure_dirs()
    x_df = load_csv(paths.X_ZSCORED_WITH_IDS_CSV)
    clusters = load_csv(paths.CLUSTER_LABELS_K2_CSV)
    df = x_df.merge(clusters, on=PATIENT_ID_COL, how="inner")
    logger.info("Merged radiomics + clusters: %s", df.shape)

    feature_cols = [c for c in df.columns if c not in (PATIENT_ID_COL, CLUSTER_COL)]
    results = compare_features(df, feature_cols)
    save_csv(results, paths.TOP_FEATURES_CSV)

    logger.info(
        "Top %d cluster-separating features:\n%s",
        N_TOP_FEATURES_DISPLAY,
        results.head(N_TOP_FEATURES_DISPLAY)[
            ["feature", "cohens_d", "p_value", "q_value"]
        ].to_string(index=False),
    )
    return results


if __name__ == "__main__":
    main()
