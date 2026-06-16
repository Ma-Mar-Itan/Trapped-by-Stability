"""Shared 'analysis frame' for the harmonization stage.

The original scripts 10 and 11 relied on a chain of in-memory variables (``df``,
``X``, ``feature_cols``, ``raw_labels``) built once and reused across scripts.
This module rebuilds that frame deterministically and explicitly, anchored to
the z-scored feature-matrix row order so that the raw phenotype labels,
manufacturer codes, and feature values are always aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from radiomics_stability import paths
from radiomics_stability.config import (
    CLUSTER_COL,
    MANUFACTURER_CLEAN_COL,
    MANUFACTURER_COL,
    PATIENT_ID_COL,
)
from radiomics_stability.io_utils import load_csv
from radiomics_stability.logging_utils import get_logger
from radiomics_stability.transforms import clean_manufacturer

logger = get_logger(__name__)


@dataclass
class AnalysisFrame:
    """Aligned inputs for harmonization analyses."""

    df: pd.DataFrame              # full frame: features + manufacturer + cluster
    x: pd.DataFrame               # feature matrix only (rows match df)
    feature_cols: list[str]       # feature column names
    raw_labels: np.ndarray        # baseline k=2 cluster labels
    manufacturer: pd.Series       # cleaned manufacturer labels


def build_harmonization_frame() -> AnalysisFrame:
    """Assemble the manufacturer-aligned analysis frame.

    Left-joins the z-scored feature matrix with manufacturer (from the merged
    dataset) and the baseline cluster labels, preserving feature-matrix row
    order.
    """
    x_df = load_csv(paths.X_ZSCORED_WITH_IDS_CSV)
    merged = load_csv(paths.MERGED_DATASET_CSV)
    raw_clusters = load_csv(paths.CLUSTER_LABELS_K2_CSV)

    df = x_df.merge(
        merged[[PATIENT_ID_COL, MANUFACTURER_COL]], on=PATIENT_ID_COL, how="inner"
    ).merge(raw_clusters, on=PATIENT_ID_COL, how="inner")
    df = clean_manufacturer(df, MANUFACTURER_COL)
    logger.info("Harmonization frame: %s", df.shape)
    logger.info("Manufacturer counts: %s", df[MANUFACTURER_CLEAN_COL].value_counts().to_dict())

    feature_cols = [c for c in x_df.columns if c != PATIENT_ID_COL]
    return AnalysisFrame(
        df=df,
        x=df[feature_cols].copy(),
        feature_cols=feature_cols,
        raw_labels=df[CLUSTER_COL].values,
        manufacturer=df[MANUFACTURER_CLEAN_COL],
    )
