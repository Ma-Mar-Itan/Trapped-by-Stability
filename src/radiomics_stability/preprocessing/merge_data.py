"""Merge the three raw TCIA tables into a single patient-level dataset.

Migrated from ``Join all data sets.py``. Behaviour is unchanged: an inner join
of imaging features, clinical metadata, and annotation boxes on ``Patient ID``.
"""

from __future__ import annotations

import pandas as pd

from radiomics_stability import paths
from radiomics_stability.config import PATIENT_ID_COL
from radiomics_stability.io_utils import load_csv, save_csv
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def merge_raw_datasets(
    clinical: pd.DataFrame,
    imaging: pd.DataFrame,
    annotation: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join imaging, clinical and annotation tables on ``Patient ID``.

    The left table is ``imaging`` so the row order matches the radiomic feature
    matrix produced downstream -- this ordering is relied upon by later stages.
    """
    merged = (
        imaging.merge(clinical, on=PATIENT_ID_COL, how="inner")
        .merge(annotation, on=PATIENT_ID_COL, how="inner")
    )
    logger.info("Merged dataset shape: %s", merged.shape)
    return merged


def main() -> pd.DataFrame:
    """Load raw CSVs, merge them, and persist the merged dataset."""
    paths.ensure_dirs()
    clinical = load_csv(paths.CLINICAL_CSV)
    imaging = load_csv(paths.IMAGING_FEATURES_CSV)
    annotation = load_csv(paths.ANNOTATION_BOXES_CSV)

    merged = merge_raw_datasets(clinical, imaging, annotation)
    save_csv(merged, paths.MERGED_DATASET_CSV)
    return merged


if __name__ == "__main__":
    main()
