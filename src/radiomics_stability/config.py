"""Central configuration: every scientific constant the pipeline depends on.

This module is the single source of truth for seeds, hyperparameters,
thresholds, label mappings, and dataset column names. Nothing here should be
changed without expecting the numerical results to change -- the values below
reproduce the manuscript outputs exactly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
#: Global random seed used for PCA, KMeans, and bootstrap resampling.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Preprocessing (see preprocessing/prepare_radiomics.py)
# ---------------------------------------------------------------------------
#: Drop features whose missing-value rate is >= this threshold.
MISSING_RATE_THRESHOLD: float = 0.05
#: Drop features with fewer than this many unique (non-NA) values.
MIN_UNIQUE_VALUES: int = 2
#: Strategy passed to sklearn.impute.SimpleImputer.
IMPUTER_STRATEGY: str = "median"

# ---------------------------------------------------------------------------
# PCA (see clustering/pca.py)
# ---------------------------------------------------------------------------
#: Solver passed to sklearn.decomposition.PCA.
PCA_SVD_SOLVER: str = "randomized"
#: Number of principal components fed into clustering.
N_PCS_CLUSTERING: int = 10
#: Number of leading PCs persisted to the PCA scores artifact (PC1..PC5).
N_PCS_SAVED: int = 5

# ---------------------------------------------------------------------------
# KMeans clustering (see clustering/kmeans.py)
# ---------------------------------------------------------------------------
#: Final number of clusters (the discovered 2-phenotype solution).
N_CLUSTERS: int = 2
#: n_init passed to sklearn.cluster.KMeans.
KMEANS_N_INIT: int = 20
#: Range of k swept during model selection (k = 2..6 inclusive).
K_RANGE: range = range(2, 7)

# ---------------------------------------------------------------------------
# Bootstrap stability (see clustering/stability.py)
# ---------------------------------------------------------------------------
#: Number of bootstrap resampling iterations.
N_BOOTSTRAPS: int = 20
#: Fraction of samples drawn (without replacement) in each bootstrap subsample.
SUBSAMPLE_FRACTION: float = 0.80

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
#: Multiple-testing correction method (statsmodels multipletests).
FDR_METHOD: str = "fdr_bh"
#: Number of top cluster-separating features echoed to the log.
N_TOP_FEATURES_DISPLAY: int = 30

# ---------------------------------------------------------------------------
# Survival analysis (see statistics/survival.py)
# ---------------------------------------------------------------------------
#: Penalizer passed to lifelines.CoxPHFitter.
COX_PENALIZER: float = 0.01
#: Days-per-year conversion used to express RFS time in years.
DAYS_PER_YEAR: float = 365.25

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
#: Resolution for saved figures.
FIGURE_DPI: int = 300

# ---------------------------------------------------------------------------
# Dataset column names (verbatim, including historical typos / trailing spaces)
# ---------------------------------------------------------------------------
PATIENT_ID_COL: str = "Patient ID"
CLUSTER_COL: str = "Cluster"
MANUFACTURER_COL: str = "Manufacturer"
MANUFACTURER_CLEAN_COL: str = "Manufacturer_clean"
SUBTYPE_COL: str = "Mol Subtype"
EVENT_COL: str = "Recurrence event(s)"

# Survival timing columns. NOTE: the trailing spaces and the "assemssment"
# typo are present in the source data and MUST be preserved to match it.
LOCAL_RECUR_COL: str = "Days to local recurrence (from the date of diagnosis) "
DISTANT_RECUR_COL: str = "Days to distant recurrence(from the date of diagnosis) "
LAST_LOCAL_COL: str = (
    "Days to last local recurrence free assessment (from the date of diagnosis) "
)
LAST_DISTANT_COL: str = (
    "Days to last distant recurrence free assemssment(from the date of diagnosis) "
)

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
#: Normalize the various manufacturer encodings to clean string labels.
#: In the released cohort only GE (0) and Siemens (2) are present.
MANUFACTURER_MAP: dict = {
    0: "GE",
    1: "MPTronic",
    2: "Siemens",
    "0": "GE",
    "1": "MPTronic",
    "2": "Siemens",
    "GE MEDICAL SYSTEMS": "GE",
    "SIEMENS": "Siemens",
    "MPTronic software": "MPTronic",
}

#: Values of ``Mol Subtype`` treated as the HR+/HER2+ subgroup. In this dataset
#: the integer code ``1`` is the matching label; the string variants are kept
#: for robustness against alternative encodings.
HRHER2_LABELS: list = [
    "HR+/HER2+",
    "HRpos_HER2pos",
    "ER/PR pos, HER2 pos",
    "ER/PR pos HER2 pos",
    1,
]
