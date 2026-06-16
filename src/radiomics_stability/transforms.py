"""Cross-cutting domain transforms shared by multiple stages.

Centralizes logic that the original scripts duplicated: manufacturer-label
cleaning (scripts 7, 9, 10, 11) and chi-square contingency testing
(``Manifcature correlation.py``, scripts 7 and 11).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency

from radiomics_stability.config import MANUFACTURER_CLEAN_COL, MANUFACTURER_MAP


def clean_manufacturer(df: pd.DataFrame, source_col: str, target_col: str = MANUFACTURER_CLEAN_COL) -> pd.DataFrame:
    """Return a copy of ``df`` with a normalized manufacturer column.

    Maps the heterogeneous manufacturer encodings (numeric codes, vendor
    strings) to clean labels via :data:`config.MANUFACTURER_MAP`.
    """
    out = df.copy()
    out[target_col] = out[source_col].replace(MANUFACTURER_MAP)
    return out


def chi2_association(df: pd.DataFrame, row_col: str, col_col: str) -> tuple[pd.DataFrame, float, float, int]:
    """Chi-square test of independence between two categorical columns.

    Returns ``(contingency_table, p_value, chi2_statistic, dof)``. The p-value
    is invariant to category relabeling, so it is identical whether raw or
    cleaned manufacturer codes are used.
    """
    table = pd.crosstab(df[row_col], df[col_col])
    chi2, p, dof, _expected = chi2_contingency(table)
    return table, float(p), float(chi2), int(dof)


def crosstab_percentages(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(row_pct, col_pct)`` for a counts contingency table.

    ``row_pct`` -- composition within each row (e.g. manufacturer mix per cluster).
    ``col_pct`` -- composition within each column (e.g. cluster mix per manufacturer).
    """
    row_pct = table.div(table.sum(axis=1), axis=0) * 100
    col_pct = table.div(table.sum(axis=0), axis=1) * 100
    return row_pct, col_pct
