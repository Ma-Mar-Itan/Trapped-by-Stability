"""radiomics_stability.

A reproducible pipeline that investigates acquisition confounding in
multi-vendor breast-MRI radiomic phenotype discovery, demonstrating that
cluster *stability* and cluster *biological identity* are independent
properties.

The package is organized into stage-oriented subpackages:

* ``preprocessing`` -- raw data merging and radiomic feature preparation
* ``clustering``    -- PCA, KMeans phenotype discovery, bootstrap stability
* ``statistics``    -- biological feature analysis, manufacturer association,
                       survival analysis
* ``harmonization`` -- manufacturer residualization, ComBat, and validation
* ``visualization`` -- shared plotting utilities

All tunable constants live in :mod:`radiomics_stability.config`; all filesystem
locations live in :mod:`radiomics_stability.paths`.
"""

from radiomics_stability import config, paths

__all__ = ["config", "paths"]
__version__ = "1.0.0"
