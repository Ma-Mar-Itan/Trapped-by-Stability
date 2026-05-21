import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# PATHS

radiomics_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\Radiomic Features Data\X_radiomics_zscored_with_ids.csv"

merged_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv"

raw_cluster_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"

output_dir = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data"


# LOAD DATA

X_df = pd.read_csv(radiomics_path)
merged = pd.read_csv(merged_path)
raw_clusters = pd.read_csv(raw_cluster_path)

df = X_df.merge(
    merged[["Patient ID", "Manufacturer"]],
    on="Patient ID",
    how="inner"
)

df = df.merge(
    raw_clusters,
    on="Patient ID",
    how="inner"
)

print("Merged shape:", df.shape)


# CLEAN MANUFACTURER

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

df["Manufacturer_clean"] = df["Manufacturer"].replace(manufacturer_map)

print(df["Manufacturer_clean"].value_counts(dropna=False))


# FEATURE MATRIX

feature_cols = [
    col for col in X_df.columns
    if col != "Patient ID"
]

X = df[feature_cols].copy()

raw_labels = df["Cluster"].values


# DESIGN MATRIX FOR MANUFACTURER

D = pd.get_dummies(
    df["Manufacturer_clean"],
    drop_first=True
).astype(float)

print("Residualization design matrix:")
print(D.head())


# LINEAR RESIDUALIZATION
# Remove manufacturer-associated linear component

lr = LinearRegression()

lr.fit(D.values, X.values)

X_pred = lr.predict(D.values)

X_resid = X.values - X_pred

X_resid = pd.DataFrame(
    X_resid,
    columns=feature_cols,
    index=df.index
)

# Re-standardize residualized features
scaler = StandardScaler()

X_resid_scaled = pd.DataFrame(
    scaler.fit_transform(X_resid),
    columns=feature_cols,
    index=df.index
)


# PCA AFTER RESIDUALIZATION

pca_resid = PCA(
    svd_solver="randomized",
    random_state=42
)

X_resid_pca = pca_resid.fit_transform(X_resid_scaled)


# KMEANS AFTER RESIDUALIZATION

kmeans_resid = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=42
)

resid_labels = kmeans_resid.fit_predict(
    X_resid_pca[:, :10]
)


# COMPARE RAW CLUSTERS VS RESIDUALIZED CLUSTERS

ari_vs_raw = adjusted_rand_score(
    raw_labels,
    resid_labels
)

print("\nARI between raw clusters and residualized clusters:")
print(ari_vs_raw)


# SAVE RESIDUALIZED CLUSTERS

resid_cluster_df = pd.DataFrame({
    "Patient ID": df["Patient ID"],
    "Raw_Cluster": raw_labels,
    "Residualized_Cluster": resid_labels,
    "Manufacturer": df["Manufacturer_clean"]
})

resid_cluster_df.to_csv(
    output_dir + r"\cluster_labels_residualized_manufacturer.csv",
    index=False
)

X_resid_scaled_with_ids = X_resid_scaled.copy()
X_resid_scaled_with_ids.insert(0, "Patient ID", df["Patient ID"].values)

X_resid_scaled_with_ids.to_csv(
    output_dir + r"\X_radiomics_residualized_manufacturer_zscored.csv",
    index=False
)

print("\nSaved:")
print(output_dir + r"\cluster_labels_residualized_manufacturer.csv")
print(output_dir + r"\X_radiomics_residualized_manufacturer_zscored.csv")



import matplotlib.pyplot as plt

plot_df = pd.DataFrame({
    "PC1_raw": X_pca[:, 0],
    "PC2_raw": X_pca[:, 1],
    "Raw_Cluster": raw_labels,
    "Residualized_Cluster": resid_labels,
    "Manufacturer": df["Manufacturer_clean"].values
})

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

# Raw clusters
for c in sorted(plot_df["Raw_Cluster"].unique()):
    temp = plot_df[plot_df["Raw_Cluster"] == c]
    axes[0].scatter(temp["PC1_raw"], temp["PC2_raw"], label=f"Cluster {c}", alpha=0.7)

axes[0].set_title("A. Raw Radiomic Clusters")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Manufacturer
for m in plot_df["Manufacturer"].dropna().unique():
    temp = plot_df[plot_df["Manufacturer"] == m]
    axes[1].scatter(temp["PC1_raw"], temp["PC2_raw"], label=m, alpha=0.7)

axes[1].set_title("B. Manufacturer")
axes[1].set_xlabel("PC1")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Residualized clusters
for c in sorted(plot_df["Residualized_Cluster"].unique()):
    temp = plot_df[plot_df["Residualized_Cluster"] == c]
    axes[2].scatter(temp["PC1_raw"], temp["PC2_raw"], label=f"Cluster {c}", alpha=0.7)

axes[2].set_title(f"C. Clusters After Manufacturer Residualization\nARI vs Raw = {ari_vs_raw:.3f}")
axes[2].set_xlabel("PC1")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

output_path = output_dir + r"\raw_manufacturer_residualized_cluster_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("Saved figure:")
print(output_path)
