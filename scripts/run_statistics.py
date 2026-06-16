"""Stage 3 -- statistics: biological characterization and manufacturer confounding.

Requires the clustering stage to have run first.

Usage:
    python scripts/run_statistics.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from radiomics_stability.statistics import biology_analysis, manufacturer_analysis


def main() -> None:
    biology_analysis.main()
    manufacturer_analysis.main()


if __name__ == "__main__":
    main()
