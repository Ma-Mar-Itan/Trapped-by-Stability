"""Recurrence-free survival (RFS) analysis in the HR+/HER2+ subgroup.

Consolidates ``6- death associated.py`` and ``9- cox model.py``, which both
rebuilt the *identical* RFS-time variables. That construction now lives in a
single :func:`build_rfs_frame` function. Provides Kaplan-Meier curves, the
log-rank test, a cluster-only Cox model, and a manufacturer-adjusted Cox model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from radiomics_stability import paths
from radiomics_stability.config import (
    CLUSTER_COL,
    COX_PENALIZER,
    DAYS_PER_YEAR,
    DISTANT_RECUR_COL,
    EVENT_COL,
    HRHER2_LABELS,
    LAST_DISTANT_COL,
    LAST_LOCAL_COL,
    LOCAL_RECUR_COL,
    MANUFACTURER_COL,
    PATIENT_ID_COL,
    SUBTYPE_COL,
)
from radiomics_stability.io_utils import load_csv, save_csv, save_figure
from radiomics_stability.logging_utils import get_logger
from radiomics_stability.transforms import clean_manufacturer

logger = get_logger(__name__)

_TIME_COL = "rfs_time_years"


def build_rfs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct recurrence-free-survival time (years) and clean invalid rows.

    Event = earliest recurrence time; censored = latest recurrence-free
    follow-up time. Rows without a valid positive RFS time are dropped. The
    recurrence-event column is coerced to numeric and restricted to {0, 1}.
    """
    out = df.copy()
    out[EVENT_COL] = pd.to_numeric(out[EVENT_COL], errors="coerce")
    out = out[out[EVENT_COL].isin([0, 1])].copy()

    for col in (LOCAL_RECUR_COL, DISTANT_RECUR_COL, LAST_LOCAL_COL, LAST_DISTANT_COL):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["recurrence_time_days"] = out[[LOCAL_RECUR_COL, DISTANT_RECUR_COL]].min(axis=1)
    out["followup_time_days"] = out[[LAST_LOCAL_COL, LAST_DISTANT_COL]].max(axis=1)
    out["rfs_time_days"] = np.where(
        out[EVENT_COL] == 1, out["recurrence_time_days"], out["followup_time_days"]
    )
    out[_TIME_COL] = out["rfs_time_days"] / DAYS_PER_YEAR

    out = out[out[_TIME_COL].notna()]
    out = out[out[_TIME_COL] > 0]
    return out


def filter_hrher2(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the HR+/HER2+ molecular subgroup."""
    sub = df[df[SUBTYPE_COL].isin(HRHER2_LABELS)].copy()
    logger.info("HR+/HER2+ sample size: %d  events: %d", sub.shape[0], int(sub[EVENT_COL].sum()))
    return sub


def _plot_km(hrher2: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    for cluster_value in sorted(hrher2[CLUSTER_COL].dropna().unique()):
        temp = hrher2[hrher2[CLUSTER_COL] == cluster_value]
        kmf.fit(
            durations=temp[_TIME_COL],
            event_observed=temp[EVENT_COL],
            label=f"Cluster {cluster_value}",
        )
        kmf.plot_survival_function(ci_show=True, ax=ax)
    ax.set_title("Recurrence-Free Survival in HR+/HER2+ by Radiomic Cluster")
    ax.set_xlabel("Time since diagnosis / MRI baseline (years)")
    ax.set_ylabel("Recurrence-free survival probability")
    ax.grid(True)
    return fig


def logrank_by_cluster(hrher2: pd.DataFrame) -> float:
    """Two-group log-rank test between the two clusters; returns the p-value."""
    values = sorted(hrher2[CLUSTER_COL].dropna().unique())
    g0 = hrher2[hrher2[CLUSTER_COL] == values[0]]
    g1 = hrher2[hrher2[CLUSTER_COL] == values[1]]
    result = logrank_test(
        g0[_TIME_COL], g1[_TIME_COL],
        event_observed_A=g0[EVENT_COL], event_observed_B=g1[EVENT_COL],
    )
    return float(result.p_value)


def outcome_summary(hrher2: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster n, event count, median RFS, and event rate."""
    summary = hrher2.groupby(CLUSTER_COL).agg(
        n=(PATIENT_ID_COL, "count"),
        events=(EVENT_COL, "sum"),
        median_rfs_years=(_TIME_COL, "median"),
    )
    summary["event_rate"] = summary["events"] / summary["n"]
    return summary


def cox_cluster_only(hrher2: pd.DataFrame) -> pd.DataFrame:
    """Cox model with cluster as the sole covariate."""
    cox_df = hrher2[[_TIME_COL, EVENT_COL, CLUSTER_COL]].dropna().copy()
    cox_df[CLUSTER_COL] = cox_df[CLUSTER_COL].astype(int)
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(cox_df, duration_col=_TIME_COL, event_col=EVENT_COL)
    return cph.summary


def cox_manufacturer_adjusted(hrher2: pd.DataFrame) -> pd.DataFrame:
    """Cox model adjusting for one-hot-encoded manufacturer."""
    cox_df = hrher2[[_TIME_COL, EVENT_COL, CLUSTER_COL, "Manufacturer_clean"]].dropna().copy()
    cox_df[CLUSTER_COL] = cox_df[CLUSTER_COL].astype(int)
    cox_df = pd.get_dummies(cox_df, columns=["Manufacturer_clean"], drop_first=True)
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(cox_df, duration_col=_TIME_COL, event_col=EVENT_COL)
    return cph.summary


def main(save_figures: bool = True) -> pd.DataFrame:
    """Run the full HR+/HER2+ survival analysis and persist summaries."""
    paths.ensure_dirs()
    merged = load_csv(paths.MERGED_DATASET_CSV)
    clusters = load_csv(paths.CLUSTER_LABELS_K2_CSV)
    df = merged.merge(clusters, on=PATIENT_ID_COL, how="inner")
    df = clean_manufacturer(df, MANUFACTURER_COL)

    df = build_rfs_frame(df)
    hrher2 = filter_hrher2(df)

    if save_figures:
        save_figure(_plot_km(hrher2), paths.FIG_KM_CURVES)

    logger.info("Log-rank p-value: %.4g", logrank_by_cluster(hrher2))

    summary = outcome_summary(hrher2)
    logger.info("Outcome summary by cluster:\n%s", summary.to_string())
    save_csv(summary, paths.SURVIVAL_SUMMARY_CSV, index=True)

    cox1 = cox_cluster_only(hrher2)
    save_csv(cox1, paths.COX_CLUSTER_ONLY_CSV, index=True)
    logger.info("Cox (cluster only):\n%s", cox1.to_string())

    cox2 = cox_manufacturer_adjusted(hrher2)
    save_csv(cox2, paths.COX_MANUFACTURER_ADJUSTED_CSV, index=True)
    logger.info("Cox (manufacturer-adjusted):\n%s", cox2.to_string())

    return summary


if __name__ == "__main__":
    main()
