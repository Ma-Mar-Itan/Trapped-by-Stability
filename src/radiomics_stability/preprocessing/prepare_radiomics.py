"""Prepare the analysis-ready radiomic feature matrix.

Migrated from ``1-Prepare Radiomics Features.py``. The QC + imputation +
standardization pipeline is unchanged:

1. Coerce all imaging-feature columns to numeric.
2. Drop features with >= ``MISSING_RATE_THRESHOLD`` missing values.
3. Drop constant / near-empty features (< ``MIN_UNIQUE_VALUES`` unique values).
4. Median-impute remaining missing values.
5. Z-score standardize.

Outputs: the z-scored matrix (with and without IDs), the patient-ID list, and
the surviving feature list.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from radiomics_stability import paths
from radiomics_stability.config import (
    IMPUTER_STRATEGY,
    MIN_UNIQUE_VALUES,
    MISSING_RATE_THRESHOLD,
    PATIENT_ID_COL,
)
from radiomics_stability.io_utils import load_csv, save_csv
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_radiomic_matrix(imaging: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Run QC, imputation and z-scoring on raw imaging features.

    Returns ``(X_scaled, patient_ids)`` where ``X_scaled`` is the standardized
    feature matrix (no ID column) aligned row-wise to ``patient_ids``.
    """
    if PATIENT_ID_COL not in imaging.columns:
        raise ValueError(f"'{PATIENT_ID_COL}' column not found in imaging features.")

    patient_ids = imaging[PATIENT_ID_COL].copy()

    # Feature matrix forced to numeric (non-parseable values -> NaN).
    x_raw = imaging.drop(columns=[PATIENT_ID_COL]).apply(pd.to_numeric, errors="coerce")
    logger.info("Raw radiomic matrix: %s", x_raw.shape)

    # QC: drop high-missing and constant features.
    missing_rate = x_raw.isna().mean()
    x_qc = x_raw.loc[:, missing_rate < MISSING_RATE_THRESHOLD]
    x_qc = x_qc.loc[:, x_qc.nunique(dropna=True) >= MIN_UNIQUE_VALUES]
    logger.info("After QC: %s", x_qc.shape)

    # Median imputation.
    imputer = SimpleImputer(strategy=IMPUTER_STRATEGY)
    x_imputed = pd.DataFrame(
        imputer.fit_transform(x_qc), columns=x_qc.columns, index=x_qc.index
    )

    # Z-score standardization.
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(
        scaler.fit_transform(x_imputed), columns=x_imputed.columns, index=x_imputed.index
    )
    logger.info("Final radiomic matrix: %s", x_scaled.shape)

    return x_scaled, patient_ids


def _with_ids(x_scaled: pd.DataFrame, patient_ids: pd.Series) -> pd.DataFrame:
    """Prepend the Patient ID column to the feature matrix."""
    out = x_scaled.copy()
    out.insert(0, PATIENT_ID_COL, patient_ids.values)
    return out


def main() -> pd.DataFrame:
    """Build and persist all processed radiomic artifacts."""
    paths.ensure_dirs()
    imaging = load_csv(paths.IMAGING_FEATURES_CSV)

    x_scaled, patient_ids = prepare_radiomic_matrix(imaging)

    save_csv(x_scaled, paths.X_ZSCORED_CSV)
    save_csv(pd.DataFrame({PATIENT_ID_COL: patient_ids}), paths.PATIENT_IDS_CSV)
    save_csv(
        pd.DataFrame({"radiomic_feature": x_scaled.columns}), paths.FEATURE_LIST_CSV
    )

    x_with_ids = _with_ids(x_scaled, patient_ids)
    save_csv(x_with_ids, paths.X_ZSCORED_WITH_IDS_CSV)
    return x_with_ids


if __name__ == "__main__":
    main()
