"""Run the entire pipeline end to end, from raw data to final figures and tables.

    python scripts/run_full_pipeline.py

Stages (each depends on the previous):
    1. preprocessing  -- merge raw tables, build z-scored feature matrix
    2. clustering     -- PCA, KMeans phenotype, bootstrap stability
    3. statistics     -- biological characterization, manufacturer confounding
    4. survival       -- HR+/HER2+ recurrence-free survival
    5. harmonization  -- residualization, ComBat, within-vendor, validation

The harmonization stage requires neuroCombat. Pass ``--skip-harmonization`` to
run everything else if neuroCombat is unavailable.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from radiomics_stability.logging_utils import get_logger

logger = get_logger("run_full_pipeline")


def main(skip_harmonization: bool = False) -> None:
    from radiomics_stability.clustering import kmeans, pca, stability
    from radiomics_stability.preprocessing import merge_data, prepare_radiomics
    from radiomics_stability.statistics import (
        biology_analysis,
        manufacturer_analysis,
        survival,
    )

    logger.info("=== Stage 1/5: preprocessing ===")
    merge_data.main()
    prepare_radiomics.main()

    logger.info("=== Stage 2/5: clustering ===")
    pca.main()
    kmeans.main()
    stability.main()

    logger.info("=== Stage 3/5: statistics ===")
    biology_analysis.main()
    manufacturer_analysis.main()

    logger.info("=== Stage 4/5: survival ===")
    survival.main()

    if skip_harmonization:
        logger.warning("Skipping harmonization stage (--skip-harmonization).")
        return

    logger.info("=== Stage 5/5: harmonization + validation ===")
    from radiomics_stability.harmonization import validation

    validation.main()
    logger.info("Pipeline complete. Outputs written under data/results/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-harmonization",
        action="store_true",
        help="Skip the ComBat/harmonization stage (use if neuroCombat is unavailable).",
    )
    args = parser.parse_args()
    main(skip_harmonization=args.skip_harmonization)
