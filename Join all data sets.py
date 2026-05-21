import pandas as pd

# Paths
clinical = pd.read_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\raw\Clinical.csv"
)
imaging = pd.read_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\raw\Imaging_Features.csv"
)
annotation = pd.read_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\raw\Annotation_Boxes.csv"
)

#Join Data sets
df = (
    imaging
    .merge(clinical, on="Patient ID", how="inner")
    .merge(annotation, on="Patient ID", how="inner")
)

print(df.shape)
print(df.head())

df.to_csv(
    r"C:\Users\malek\Desktop\Dr. Lama\Separating Cluster Stability from Biological Identity in Breast MRI Radiomic Phenotype Discovery\data\merged_dataset.csv",
    index=False
)
