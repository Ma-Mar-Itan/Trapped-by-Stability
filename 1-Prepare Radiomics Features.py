!pip install scikit-learn
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATHS
# =========================================================
imaging_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\raw\Imaging_Features.csv"

merged_path = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv"

# =========================================================
# LOAD IMAGING FEATURES ONLY
# =========================================================
img = pd.read_csv(imaging_path)

print("Imaging shape:", img.shape)
print(img.head())

# =========================================================
# IDENTIFY PATIENT ID
# =========================================================
id_col = "Patient ID"

if id_col not in img.columns:
    raise ValueError("Patient ID column not found. Check the column names.")

patient_ids = img[id_col].copy()

# =========================================================
# EXTRACT RADIOMIC FEATURES
# =========================================================
X_raw = img.drop(columns=[id_col]).copy()

# force all feature columns to numeric
X_raw = X_raw.apply(pd.to_numeric, errors="coerce")

print("Raw radiomic matrix:", X_raw.shape)

# =========================================================
# BASIC QC
# =========================================================

# remove features with >=5% missing values
missing_rate = X_raw.isna().mean()
X_qc = X_raw.loc[:, missing_rate < 0.05]

# remove constant / near-empty features
X_qc = X_qc.loc[:, X_qc.nunique(dropna=True) >= 2]

print("After QC:", X_qc.shape)

# =========================================================
# IMPUTE MISSING VALUES
# =========================================================
imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X_qc),
    columns=X_qc.columns,
    index=X_qc.index
)

# =========================================================
# Z-SCORE STANDARDIZATION
# =========================================================
scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)

print("Final radiomic matrix:", X_scaled.shape)

# =========================================================
# SAVE OUTPUTS
# =========================================================
output_dir = r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data"

X_scaled.to_csv(
    output_dir + r"\X_radiomics_zscored.csv",
    index=False
)

pd.DataFrame({"Patient ID": patient_ids}).to_csv(
    output_dir + r"\patient_ids.csv",
    index=False
)

feature_list = pd.DataFrame({"radiomic_feature": X_scaled.columns})
feature_list.to_csv(
    output_dir + r"\radiomic_feature_list.csv",
    index=False
)

print("Saved:")
print(output_dir + r"\X_radiomics_zscored.csv")
print(output_dir + r"\patient_ids.csv")
print(output_dir + r"\radiomic_feature_list.csv")


# =========================================================
# ADD PATIENT IDs BACK
# =========================================================

X_scaled_with_ids = X_scaled.copy()

X_scaled_with_ids.insert(
    0,
    "Patient ID",
    patient_ids.values
)

print(X_scaled_with_ids.shape)
print(X_scaled_with_ids.head())

# =========================================================
# SAVE VERSION WITH IDs
# =========================================================

X_scaled_with_ids.to_csv(
    output_dir + r"\X_radiomics_zscored_with_ids.csv",
    index=False
)

print("Saved:")
print(output_dir + r"\X_radiomics_zscored_with_ids.csv")
