import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# LOAD CLEANED RADIOOMIC MATRIX

path = (
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\Radiomic Features Data\X_radiomics_zscored_with_ids.csv"
)

df = pd.read_csv(path)

print(df.shape)


# SEPARATE IDs FROM FEATURES

patient_ids = df["Patient ID"]

X = df.drop(columns=["Patient ID"])

print("Feature matrix:", X.shape)


# RUN PCA

pca = PCA(
    svd_solver="randomized",
    random_state=42
)

X_pca = pca.fit_transform(X)


# EXPLAINED VARIANCE

explained_variance = pca.explained_variance_ratio_

print("\nTop 10 PCs explained variance:")
for i in range(10):
    print(f"PC{i+1}: {explained_variance[i]:.4f}")


# CREATE PCA DATAFRAME

pca_df = pd.DataFrame({
    "Patient ID": patient_ids,
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "PC3": X_pca[:, 2],
    "PC4": X_pca[:, 3],
    "PC5": X_pca[:, 4]
})

print(pca_df.head())


# SAVE PCA SCORES
output_path = (
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\PCA_scores.csv"
)

pca_df.to_csv(output_path, index=False)

print("\nSaved PCA scores:")
print(output_path)


# SCREE PLOT

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, 21),
    explained_variance[:20],
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Top 20 Principal Components")

plt.grid(True)
plt.show()

# PC1 vs PC2 SCATTER
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection: PC1 vs PC2")

plt.grid(True)
plt.show()
