"""Stage 5 -- harmonization + validation synthesis.

Runs manufacturer residualization, ComBat harmonization, within-vendor
clustering, post-harmonization association tests, and cross-condition bootstrap
stability. Requires the clustering stage to have run first and neuroCombat to be
installed.

Usage:
    python scripts/run_harmonization.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from radiomics_stability.harmonization import validation


def main() -> None:
    # validation.main() builds the shared frame once and drives residualization,
    # ComBat, within-vendor clustering, association tests, and stability.
    validation.main()


if __name__ == "__main__":
    main()
