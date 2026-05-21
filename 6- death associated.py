import sys
!"{sys.executable}" -m pip install lifelines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# =========================================================
# PATHS
# =========================================================
merged_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv"

cluster_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"

# =========================================================
# LOAD + MERGE
# =========================================================
merged = pd.read_csv(merged_path)
clusters = pd.read_csv(cluster_path)

df = merged.merge(clusters, on="Patient ID", how="inner")

print(df.shape)

# =========================================================
# COLUMN NAMES
# =========================================================
subtype_col = "Mol Subtype"
event_col = "Recurrence event(s)"

local_recur_col = "Days to local recurrence (from the date of diagnosis) "
distant_recur_col = "Days to distant recurrence(from the date of diagnosis) "
last_local_col = "Days to last local recurrence free assessment (from the date of diagnosis) "
last_distant_col = "Days to last distant recurrence free assemssment(from the date of diagnosis) "

# =========================================================
# CLEAN EVENT VARIABLE
# =========================================================
df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

# Keep only valid 0/1 recurrence events
df = df[df[event_col].isin([0, 1])].copy()

# =========================================================
# BUILD RFS TIME
# If recurrence = 1: use earliest recurrence time
# If recurrence = 0: use latest recurrence-free follow-up time
# =========================================================
for col in [local_recur_col, distant_recur_col, last_local_col, last_distant_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["recurrence_time_days"] = df[[local_recur_col, distant_recur_col]].min(axis=1)
df["followup_time_days"] = df[[last_local_col, last_distant_col]].max(axis=1)

df["rfs_time_days"] = np.where(
    df[event_col] == 1,
    df["recurrence_time_days"],
    df["followup_time_days"]
)

df["rfs_time_years"] = df["rfs_time_days"] / 365.25

# Remove invalid times
df = df[df["rfs_time_years"].notna()]
df = df[df["rfs_time_years"] > 0]

# =========================================================
# CHECK SUBTYPE VALUES
# =========================================================
print("\nSubtype values:")
print(df[subtype_col].value_counts(dropna=False))

# =========================================================
# FILTER HR+/HER2+
# Adjust this value if your file uses a slightly different label
# =========================================================
hrher2_labels = [
    "HR+/HER2+",
    "HRpos_HER2pos",
    "ER/PR pos, HER2 pos",
    "ER/PR pos HER2 pos",
    1
]

hrher2 = df[df[subtype_col].isin(hrher2_labels)].copy()

print("\nHR+/HER2+ sample size:", hrher2.shape[0])
print("Events:", int(hrher2[event_col].sum()))
print(hrher2["Cluster"].value_counts())

# =========================================================
# KAPLAN-MEIER CURVES BY CLUSTER
# =========================================================
kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))

for cluster_value in sorted(hrher2["Cluster"].dropna().unique()):

    temp = hrher2[hrher2["Cluster"] == cluster_value]

    kmf.fit(
        durations=temp["rfs_time_years"],
        event_observed=temp[event_col],
        label=f"Cluster {cluster_value}"
    )

    kmf.plot_survival_function(ci_show=True)

plt.title("Recurrence-Free Survival in HR+/HER2+ by Radiomic Cluster")
plt.xlabel("Time since diagnosis / MRI baseline (years)")
plt.ylabel("Recurrence-free survival probability")
plt.grid(True)
plt.show()

# =========================================================
# LOG-RANK TEST
# =========================================================
cluster_values = sorted(hrher2["Cluster"].dropna().unique())

group0 = hrher2[hrher2["Cluster"] == cluster_values[0]]
group1 = hrher2[hrher2["Cluster"] == cluster_values[1]]

logrank = logrank_test(
    group0["rfs_time_years"],
    group1["rfs_time_years"],
    event_observed_A=group0[event_col],
    event_observed_B=group1[event_col]
)

print("\nLog-rank test:")
print("p-value:", logrank.p_value)

# =========================================================
# EVENT RATE BY CLUSTER
# =========================================================
event_summary = hrher2.groupby("Cluster").agg(
    n=("Patient ID", "count"),
    events=(event_col, "sum"),
    median_rfs_years=("rfs_time_years", "median")
)

event_summary["event_rate"] = event_summary["events"] / event_summary["n"]

print("\nOutcome summary by cluster:")
print(event_summary)

# =========================================================
# COX MODEL: CLUSTER ONLY
# =========================================================
cox_df = hrher2[["rfs_time_years", event_col, "Cluster"]].dropna().copy()

cox_df["Cluster"] = cox_df["Cluster"].astype(int)

cph = CoxPHFitter(penalizer=0.01)

cph.fit(
    cox_df,
    duration_col="rfs_time_years",
    event_col=event_col
)

print("\nCox model: cluster only")
cph.print_summary()
