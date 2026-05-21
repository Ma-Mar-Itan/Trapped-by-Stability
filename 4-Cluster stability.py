import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

n_bootstraps = 20
subsample_fraction = 0.80

# =========================================================
# BASELINE CLUSTERING
# =========================================================

X_cluster = X_pca[:, :10]

baseline_kmeans = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=RANDOM_SEED
)

baseline_labels = baseline_kmeans.fit_predict(X_cluster)

# =========================================================
# BOOTSTRAP STABILITY
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

ari_scores = []

n_samples = X_cluster.shape[0]

for b in range(n_bootstraps):

    # ---------------------------------------------
    # random 80% subsample
    # ---------------------------------------------
    subset_idx = rng.choice(
        n_samples,
        size=int(subsample_fraction * n_samples),
        replace=False
    )

    X_subset = X_cluster[subset_idx]

    # ---------------------------------------------
    # recluster subset
    # ---------------------------------------------
    kmeans_boot = KMeans(
        n_clusters=2,
        n_init=20,
        random_state=RANDOM_SEED + b
    )

    subset_labels = kmeans_boot.fit_predict(X_subset)

    # ---------------------------------------------
    # compare against original clustering
    # ---------------------------------------------
    baseline_subset_labels = baseline_labels[subset_idx]

    ari = adjusted_rand_score(
        baseline_subset_labels,
        subset_labels
    )

    ari_scores.append(ari)

# =========================================================
# RESULTS
# =========================================================

ari_scores = np.array(ari_scores)

print("\nBootstrap ARI scores:")
print(np.round(ari_scores, 4))

print("\nMean ARI:", round(ari_scores.mean(), 4))
print("Std ARI :", round(ari_scores.std(), 4))

# =========================================================
# SAVE RESULTS
# =========================================================

ari_df = pd.DataFrame({
    "bootstrap": np.arange(1, n_bootstraps + 1),
    "ARI": ari_scores
})

output_path = (
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\bootstrap_ari_results.csv"
)

ari_df.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)

# =========================================================
# VISUALIZE
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    np.arange(1, n_bootstraps + 1),
    ari_scores,
    marker="o"
)

plt.axhline(
    ari_scores.mean(),
    linestyle="--"
)

plt.xlabel("Bootstrap Iteration")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Bootstrap Stability of k=2 Phenotype")

plt.grid(True)

plt.show()
