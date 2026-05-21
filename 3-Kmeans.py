import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# =========================================================
# USE PCA SPACE FOR CLUSTERING
# =========================================================

# use first 10 PCs
X_cluster = X_pca[:, :10]

print("Clustering matrix shape:", X_cluster.shape)

# =========================================================
# TEST k = 2 TO 6
# =========================================================

results = []

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42
    )

    labels = kmeans.fit_predict(X_cluster)

    silhouette = silhouette_score(X_cluster, labels)

    ch_score = calinski_harabasz_score(
        X_cluster,
        labels
    )

    db_score = davies_bouldin_score(
        X_cluster,
        labels
    )

    results.append({
        "k": k,
        "silhouette": silhouette,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score
    })

# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\nCluster validity results:")
print(results_df)

# =========================================================
# PLOT METRICS
# =========================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# silhouette
axes[0].plot(
    results_df["k"],
    results_df["silhouette"],
    marker="o"
)

axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("k")

# CH
axes[1].plot(
    results_df["k"],
    results_df["calinski_harabasz"],
    marker="o"
)

axes[1].set_title("Calinski-Harabasz")
axes[1].set_xlabel("k")

# DB
axes[2].plot(
    results_df["k"],
    results_df["davies_bouldin"],
    marker="o"
)

axes[2].set_title("Davies-Bouldin")
axes[2].set_xlabel("k")

plt.tight_layout()
plt.show()

# =========================================================
# FIT FINAL k=2 MODEL
# =========================================================

kmeans_final = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=42
)

cluster_labels = kmeans_final.fit_predict(X_cluster)

# =========================================================
# SAVE CLUSTER LABELS
# =========================================================

cluster_df = pd.DataFrame({
    "Patient ID": patient_ids,
    "Cluster": cluster_labels
})

output_path = (
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\cluster_labels_k2.csv"
)

cluster_df.to_csv(output_path, index=False)

print("\nSaved cluster labels:")
print(output_path)

# =========================================================
# VISUALIZE CLUSTERS
# =========================================================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters (k=2)")

plt.grid(True)

plt.show()
