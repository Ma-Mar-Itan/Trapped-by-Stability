import pandas as pd
import numpy as np

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# =========================================================
# PATHS
# =========================================================
radiomics_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\Radiomic Features Data\X_radiomics_zscored_with_ids.csv"

cluster_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"

output_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\top_cluster_features.csv"

# =========================================================
# LOAD DATA
# =========================================================
X_df = pd.read_csv(radiomics_path)
clusters = pd.read_csv(cluster_path)

df = X_df.merge(
    clusters,
    on="Patient ID",
    how="inner"
)

print("Merged radiomics + clusters:", df.shape)

# =========================================================
# FEATURE COLUMNS
# =========================================================
feature_cols = [
    col for col in df.columns
    if col not in ["Patient ID", "Cluster"]
]

# =========================================================
# SPLIT BY CLUSTER
# =========================================================
cluster0 = df[df["Cluster"] == 0]
cluster1 = df[df["Cluster"] == 1]

print("Cluster 0 n:", cluster0.shape[0])
print("Cluster 1 n:", cluster1.shape[0])

# =========================================================
# FEATURE-WISE DIFFERENCE TESTS
# =========================================================
results = []

for feature in feature_cols:

    x0 = cluster0[feature].dropna()
    x1 = cluster1[feature].dropna()

    mean0 = x0.mean()
    mean1 = x1.mean()

    sd0 = x0.std(ddof=1)
    sd1 = x1.std(ddof=1)

    n0 = len(x0)
    n1 = len(x1)

    # Welch t-test
    t_stat, p_value = ttest_ind(
        x0,
        x1,
        equal_var=False,
        nan_policy="omit"
    )

    # Cohen's d
    pooled_sd = np.sqrt(
        ((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2)
        / (n0 + n1 - 2)
    )

    if pooled_sd == 0:
        cohens_d = np.nan
    else:
        cohens_d = (mean1 - mean0) / pooled_sd

    results.append({
        "feature": feature,
        "mean_cluster0": mean0,
        "mean_cluster1": mean1,
        "difference_cluster1_minus_cluster0": mean1 - mean0,
        "cohens_d": cohens_d,
        "abs_cohens_d": abs(cohens_d),
        "p_value": p_value
    })

results_df = pd.DataFrame(results)

# =========================================================
# MULTIPLE TESTING CORRECTION
# =========================================================
results_df["q_value"] = multipletests(
    results_df["p_value"],
    method="fdr_bh"
)[1]

# =========================================================
# SORT BY EFFECT SIZE
# =========================================================
results_df = results_df.sort_values(
    by="abs_cohens_d",
    ascending=False
)

# =========================================================
# SAVE RESULTS
# =========================================================
results_df.to_csv(output_path, index=False)

print("\nSaved top cluster-separating features:")
print(output_path)

# =========================================================
# DISPLAY TOP 30 FEATURES
# =========================================================
top30 = results_df.head(30)

print("\nTop 30 cluster-separating radiomic features:")
print(
    top30[
        [
            "feature",
            "mean_cluster0",
            "mean_cluster1",
            "difference_cluster1_minus_cluster0",
            "cohens_d",
            "p_value",
            "q_value"
        ]
    ]
)
