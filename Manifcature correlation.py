import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency, fisher_exact


# PATHS

merged_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv"

cluster_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"


# LOAD DATA

merged = pd.read_csv(merged_path)
clusters = pd.read_csv(cluster_path)

print("Merged shape:", merged.shape)
print("Clusters shape:", clusters.shape)


# MERGE CLUSTER LABELS INTO FULL DATASET

df_clustered = merged.merge(
    clusters,
    on="Patient ID",
    how="inner"
)

print("Clustered dataset shape:", df_clustered.shape)


# CHECK COLUMN NAMES

print("\nColumns containing possible manufacturer/subtype/recurrence info:")

for col in df_clustered.columns:
    col_lower = col.lower()
    if (
        "manufacturer" in col_lower
        or "subtype" in col_lower
        or "recurrence" in col_lower
        or "event" in col_lower
    ):
        print(col)

manufacturer_col = "Manufacturer"
subtype_col = "Mol Subtype"
recurrence_col = "Recurrence event(s)"


# SET COLUMN NAMES


manufacturer_col = "Manufacturer"
subtype_col = "Mol Subtype"
recurrence_col = "Recurrence event(s)"


# ASSOCIATION TEST FUNCTION


def chi_square_test(data, row_col, col_col):

    table = pd.crosstab(
        data[row_col],
        data[col_col]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    print("\n========================================")
    print(f"{row_col} vs {col_col}")
    print("========================================")

    print("\nContingency table:")
    print(table)

    print("\nChi-square p-value:", p)

    return table, p


# CLUSTER vs MANUFACTURER


manufacturer_table, manufacturer_p = chi_square_test(
    df_clustered,
    "Cluster",
    manufacturer_col
)


# CLUSTER vs SUBTYPE


subtype_table, subtype_p = chi_square_test(
    df_clustered,
    "Cluster",
    subtype_col
)


# CLUSTER vs RECURRENCE


recurrence_table, recurrence_p = chi_square_test(
    df_clustered,
    "Cluster",
    recurrence_col
)


# SUMMARY TABLE


association_results = pd.DataFrame({
    "Variable": [
        "Manufacturer",
        "Molecular Subtype",
        "Recurrence"
    ],
    "P-value": [
        manufacturer_p,
        subtype_p,
        recurrence_p
    ]
})

print("\n========================================")
print("ASSOCIATION SUMMARY")
print("========================================")

print(association_results)


# SAVE RESULTS


output_path = (
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_association_results.csv"
)

association_results.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)
