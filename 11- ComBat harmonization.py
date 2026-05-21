import sys
!"{sys.executable}" -m pip install neuroCombat

import pandas as pd
import numpy as np

from neuroCombat import neuroCombat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# COMBAT HARMONIZATION


covars = pd.DataFrame({
    "Manufacturer": df["Manufacturer_clean"].values
})

combat_result = neuroCombat(
    dat=X.T,
    covars=covars,
    batch_col="Manufacturer"
)

X_combat = pd.DataFrame(
    combat_result["data"].T,
    columns=feature_cols,
    index=df.index
)

X_combat_scaled = pd.DataFrame(
    StandardScaler().fit_transform(X_combat),
    columns=feature_cols,
    index=df.index
)

pca_combat = PCA(
    svd_solver="randomized",
    random_state=42
)

X_combat_pca = pca_combat.fit_transform(X_combat_scaled)

kmeans_combat = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=42
)

combat_labels = kmeans_combat.fit_predict(X_combat_pca[:, :10])

combat_ari_vs_raw = adjusted_rand_score(raw_labels, combat_labels)

print("ComBat ARI vs raw:", combat_ari_vs_raw)

combat_cluster_df = pd.DataFrame({
    "Patient ID": df["Patient ID"],
    "Raw_Cluster": raw_labels,
    "ComBat_Cluster": combat_labels,
    "Manufacturer": df["Manufacturer_clean"]
})

combat_cluster_df.to_csv(
    output_dir + r"\cluster_labels_combat.csv",
    index=False
)


###Within-manufacturer clustering
from sklearn.metrics import silhouette_score

within_results = []

for manufacturer in df["Manufacturer_clean"].unique():

    idx = df["Manufacturer_clean"] == manufacturer

    X_sub = X.loc[idx].copy()
    raw_sub = raw_labels[idx]

    X_sub_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_sub),
        columns=feature_cols,
        index=X_sub.index
    )

    pca_sub = PCA(
        svd_solver="randomized",
        random_state=42
    )

    X_sub_pca = pca_sub.fit_transform(X_sub_scaled)

    kmeans_sub = KMeans(
        n_clusters=2,
        n_init=20,
        random_state=42
    )

    sub_labels = kmeans_sub.fit_predict(X_sub_pca[:, :10])

    ari_sub = adjusted_rand_score(raw_sub, sub_labels)
    sil_sub = silhouette_score(X_sub_pca[:, :10], sub_labels)

    within_results.append({
        "Manufacturer": manufacturer,
        "n": idx.sum(),
        "ARI_vs_raw": ari_sub,
        "Silhouette": sil_sub
    })

    pd.DataFrame({
        "Patient ID": df.loc[idx, "Patient ID"],
        "Raw_Cluster": raw_sub,
        "Within_Manufacturer_Cluster": sub_labels,
        "Manufacturer": manufacturer
    }).to_csv(
        output_dir + rf"\cluster_labels_within_{manufacturer}.csv",
        index=False
    )

within_results_df = pd.DataFrame(within_results)

print(within_results_df)

within_results_df.to_csv(
    output_dir + r"\within_manufacturer_clustering_results.csv",
    index=False
)

#Biological/outcome association after harmonization
from scipy.stats import chi2_contingency


# HELPER FUNCTION


def association_test(data, cluster_col, variable_col):
    table = pd.crosstab(data[cluster_col], data[variable_col])
    chi2, p, dof, expected = chi2_contingency(table)

    return table, p


# CREATE ANALYSIS DATAFRAME


harm_df = df.copy()

harm_df["Residualized_Cluster"] = resid_labels
harm_df["ComBat_Cluster"] = combat_labels


# TEST ASSOCIATIONS


# ADD CLINICAL COLUMNS INTO harm_df


clinical_cols = [
    "Patient ID",
    "Mol Subtype",
    "Recurrence event(s)"
]

harm_df = harm_df.merge(
    merged[clinical_cols],
    on="Patient ID",
    how="left"
)

print(harm_df[["Mol Subtype", "Recurrence event(s)"]].head())
print(harm_df.columns[-10:])
tests = []

for cluster_col in ["Cluster", "Residualized_Cluster", "ComBat_Cluster"]:

    for variable_col in [
        "Manufacturer_clean",
        "Mol Subtype",
        "Recurrence event(s)"
    ]:

        table, p = association_test(
            harm_df,
            cluster_col,
            variable_col
        )

        tests.append({
            "Cluster_Type": cluster_col,
            "Variable": variable_col,
            "p_value": p
        })

        print("\n===================================")
        print(cluster_col, "vs", variable_col)
        print("===================================")
        print(table)
        print("p =", p)

association_after_harmonization = pd.DataFrame(tests)

association_after_harmonization.to_csv(
    output_dir + r"\association_after_harmonization.csv",
    index=False
)

print("\nSummary:")
print(association_after_harmonization)


#Bootstrap stability after harmonization
from sklearn.metrics import adjusted_rand_score

def bootstrap_ari(X_matrix, base_labels, n_bootstraps=20, frac=0.80, seed=42):

    rng = np.random.default_rng(seed)

    scores = []
    n = X_matrix.shape[0]

    for b in range(n_bootstraps):

        idx = rng.choice(
            n,
            size=int(frac * n),
            replace=False
        )

        X_sub = X_matrix[idx]

        labels_sub = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=seed + b
        ).fit_predict(X_sub)

        ari = adjusted_rand_score(
            base_labels[idx],
            labels_sub
        )

        scores.append(ari)

    return np.array(scores)



# RAW STABILITY

raw_boot = bootstrap_ari(
    X_pca[:, :10],
    raw_labels
)


# RESIDUALIZED STABILITY

resid_boot = bootstrap_ari(
    X_resid_pca[:, :10],
    resid_labels
)


# COMBAT STABILITY

combat_boot = bootstrap_ari(
    X_combat_pca[:, :10],
    combat_labels
)

stability_summary = pd.DataFrame({
    "Condition": ["Raw", "Residualized", "ComBat"],
    "Mean_ARI": [
        raw_boot.mean(),
        resid_boot.mean(),
        combat_boot.mean()
    ],
    "SD_ARI": [
        raw_boot.std(),
        resid_boot.std(),
        combat_boot.std()
    ],
    "ARI_vs_raw": [
        1.0,
        adjusted_rand_score(raw_labels, resid_labels),
        adjusted_rand_score(raw_labels, combat_labels)
    ]
})

print(stability_summary)

stability_summary.to_csv(
    output_dir + r"\stability_summary_raw_resid_combat.csv",
    index=False
)


#final comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.bar(
    stability_summary["Condition"],
    stability_summary["Mean_ARI"],
    yerr=stability_summary["SD_ARI"],
    capsize=5
)

plt.ylabel("Bootstrap ARI")
plt.title("Cluster Stability Across Raw and Harmonized Conditions")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir + r"\stability_raw_resid_combat.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
