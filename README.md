# Duke Breast MRI Radiomics Acquisition-Confounding Reanalysis

## Inputs
Place the three TCIA/Saha CSVs in `data/raw/` with these names:

- `Imaging_Features(Imaging Features).csv`
- `Clinical_and_Other_Features(Data).csv`
- `Annotation_Boxes(Sheet1).csv`

The uploaded files have already been copied into this folder in the generated project archive.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/09_run_all.py
```

## Output
Tables are written to `outputs/tables/`; figures are written to `outputs/figures/`; merged analysis files and fold-level predictions are written to `outputs/derived/`.

## Terminology
Use "repeated 5-fold CV" unless you add an explicit inner loop for tuning. The supplied prediction script fits all preprocessing inside training folds, including imputation, scaling, residualization, feature selection, PCA score construction, and ComBat via `neuroHarmonize`.
