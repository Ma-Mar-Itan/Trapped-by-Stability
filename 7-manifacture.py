import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# PATHS

merged_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv"
cluster_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"


# LOAD + MERGE

merged = pd.read_csv(merged_path)
clusters = pd.read_csv(cluster_path)

df = merged.merge(clusters, on="Patient ID", how="inner")

manufacturer_col = "Manufacturer"

# Optional: clean manufacturer labels if encoded numerically
manufacturer_map = {
    0: "GE",
    1: "MPTronic",
    2: "Siemens",
    "0": "GE",
    "1": "MPTronic",
    "2": "Siemens",
    "GE MEDICAL SYSTEMS": "GE",
    "SIEMENS": "Siemens",
    "MPTronic software": "MPTronic"
}

df["Manufacturer_clean"] = df[manufacturer_col].replace(manufacturer_map)


# CONTINGENCY TABLE

table = pd.crosstab(
    df["Cluster"],
    df["Manufacturer_clean"]
)

print("\nCluster × Manufacturer table:")
print(table)

chi2, p, dof, expected = chi2_contingency(table)

print("\nChi-square test:")
print("chi2:", chi2)
print("dof:", dof)
print("p-value:", p)

# Row percentages: within each cluster, what % is each manufacturer?
row_pct = table.div(table.sum(axis=1), axis=0) * 100

print("\nRow percentages:")
print(row_pct.round(2))

# Column percentages: within each manufacturer, what % goes to each cluster?
col_pct = table.div(table.sum(axis=0), axis=1) * 100

print("\nColumn percentages:")
print(col_pct.round(2))


# PLOT 1: COUNTS

plt.figure(figsize=(8, 6))

table.plot(
    kind="bar",
    figsize=(8, 6)
)

plt.title("Cluster Membership by MRI Manufacturer")
plt.xlabel("Radiomic Cluster")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.legend(title="Manufacturer")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# PLOT 2: ROW-NORMALIZED PERCENTAGES

plt.figure(figsize=(8, 6))

row_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.title("Manufacturer Composition Within Each Radiomic Cluster")
plt.xlabel("Radiomic Cluster")
plt.ylabel("Percentage of Patients")
plt.xticks(rotation=0)
plt.legend(title="Manufacturer", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# PLOT 3: COLUMN-NORMALIZED PERCENTAGES

plt.figure(figsize=(8, 6))

col_pct.T.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.title("Radiomic Cluster Distribution Within Each Manufacturer")
plt.xlabel("MRI Manufacturer")
plt.ylabel("Percentage of Patients")
plt.xticks(rotation=0)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# PLOT 4: PCA SCATTER COLORED BY MANUFACTURER
# Requires X_pca from your PCA step

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["Cluster"].values,
    "Manufacturer": df["Manufacturer_clean"].values
})

plt.figure(figsize=(9, 7))

for manufacturer in plot_df["Manufacturer"].dropna().unique():
    temp = plot_df[plot_df["Manufacturer"] == manufacturer]
    
    plt.scatter(
        temp["PC1"],
        temp["PC2"],
        label=manufacturer,
        alpha=0.7
    )

plt.title("PCA Space Colored by Manufacturer")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Manufacturer")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# PLOT 5: PCA SCATTER COLORED BY CLUSTER

plt.figure(figsize=(9, 7))

for cluster in sorted(plot_df["Cluster"].dropna().unique()):
    temp = plot_df[plot_df["Cluster"] == cluster]
    
    plt.scatter(
        temp["PC1"],
        temp["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

plt.title("PCA Space Colored by Radiomic Cluster")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############

import pandas as pd
import matplotlib.pyplot as plt

# Assumes you already have:
# X_pca
# df with Cluster and Manufacturer_clean

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["Cluster"].values,
    "Manufacturer": df["Manufacturer_clean"].values
})

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)


# LEFT: COLORED BY CLUSTER

for cluster in sorted(plot_df["Cluster"].dropna().unique()):
    temp = plot_df[plot_df["Cluster"] == cluster]

    axes[0].scatter(
        temp["PC1"],
        temp["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.7,
        s=35
    )

axes[0].set_title("A. Radiomic Clusters")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend(title="Cluster")
axes[0].grid(True, alpha=0.3)


# RIGHT: COLORED BY MANUFACTURER

for manufacturer in plot_df["Manufacturer"].dropna().unique():
    temp = plot_df[plot_df["Manufacturer"] == manufacturer]

    axes[1].scatter(
        temp["PC1"],
        temp["PC2"],
        label=manufacturer,
        alpha=0.7,
        s=35
    )

axes[1].set_title("B. MRI Manufacturer")
axes[1].set_xlabel("PC1")
axes[1].legend(title="Manufacturer")
axes[1].grid(True, alpha=0.3)


# MAIN TITLE

fig.suptitle(
    "Radiomic Phenotype and Manufacturer Overlap in PCA Space",
    fontsize=16
)

plt.tight_layout()

# Save high-resolution figure
output_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_manufacturer_overlap_pca.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("Saved figure:")
print(output_path)



# Cluster × Manufacturer table
table = pd.crosstab(df["Cluster"], df["Manufacturer_clean"])

row_pct = table.div(table.sum(axis=1), axis=0) * 100

print("Counts:")
print(table)

print("\nPercentages within cluster:")
print(row_pct.round(1))

# Save
table.to_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_by_manufacturer_counts.csv"
)

row_pct.to_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_by_manufacturer_percentages.csv"
)




import pandas as pd


# CLUSTER × MANUFACTURER SUMMARY


table = pd.crosstab(
    df["Cluster"],
    df["Manufacturer_clean"]
)

row_pct = table.div(
    table.sum(axis=1),
    axis=0
) * 100

col_pct = table.div(
    table.sum(axis=0),
    axis=1
) * 100

print("\nCluster × Manufacturer counts:")
print(table)

print("\nManufacturer percentage within each cluster:")
print(row_pct.round(1))

print("\nCluster percentage within each manufacturer:")
print(col_pct.round(1))


# SAVE TABLES


output_dir = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data"

table.to_csv(output_dir + r"\cluster_by_manufacturer_counts.csv")
row_pct.to_csv(output_dir + r"\cluster_by_manufacturer_percentages_within_cluster.csv")
col_pct.to_csv(output_dir + r"\cluster_by_manufacturer_percentages_within_manufacturer.csv")

print("\nSaved:")
print(output_dir + r"\cluster_by_manufacturer_counts.csv")
print(output_dir + r"\cluster_by_manufacturer_percentages_within_cluster.csv")
print(output_dir + r"\cluster_by_manufacturer_percentages_within_manufacturer.csv")

presentation_table = row_pct.round(1).astype(str) + "%"

print("\nPresentation table:")
print(presentation_table)

presentation_table.to_csv(
    output_dir + r"\cluster_manufacturer_presentation_table.csv"
)
