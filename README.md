# Trapped by Stability

### A Falsification Framework for Acquisition-Driven Radiomic Phenotypes


Developed as part of:

**MDPH 613 — Quality Assurance of Medical Images**
Beirut Arab University
Supervisor: Dr. Lama Affara

---

# Overview

This repository contains the full analysis pipeline, preprocessing workflow, and statistical framework used to investigate acquisition confounding in multi-vendor breast MRI radiomic phenotype discovery.

Using a cohort of 922 breast MRI patients, the study demonstrates that:

* highly stable unsupervised radiomic clusters can still be acquisition-driven rather than biologically meaningful,
* conventional clustering stability metrics alone are insufficient for validating radiomic phenotypes,
* harmonization methods can fundamentally alter cluster identity and downstream prognostic associations.

The project introduces a falsification-oriented framework designed to distinguish:

* **cluster stability**
  from
* **cluster biological validity**

---

# Main Research Question

Can a radiomic phenotype remain statistically stable while failing biological validity checks due to acquisition confounding?

This work evaluates that question through:

* multi-condition phenotype discovery,
* harmonization sensitivity testing,
* acquisition variance partitioning,
* within-vendor validation,
* vendor-clean continuous phenotype scoring.

---

# Dataset

The analysis was performed on a retrospective cohort of:

* **922 breast MRI patients**
* multi-vendor MRI acquisitions
* GE and Siemens scanners
* extracted radiomic imaging features
* clinical variables
* annotation bounding boxes


---


# Key Findings

The analysis showed that:

* the discovered phenotype was highly bootstrap-stable across conditions,
* the dominant principal components were strongly associated with acquisition parameters,
* harmonization collapsed the original cluster partition despite preserved stability,
* prognostic associations were not robust across harmonization strategies,
* post-harmonization residual structures failed predefined biomarker criteria.

The core methodological conclusion is:

> **Cluster stability and cluster biological identity are independent properties.**

---

# Reproducibility

The analysis was implemented in:

* Python 3.12
* scikit-learn
* lifelines
* statsmodels
* pandas
* numpy
* matplotlib

Random seed:

```python
RANDOM_SEED = 42
```


---

# Disclaimer

This repository is intended for research and educational purposes only.

Patient-level imaging and clinical data are not publicly distributed due to privacy and institutional restrictions.

---

# Acknowledgments

* Beirut Arab University
* MDPH 613 — QA of Medical Images
* Dr. Lama Affara


