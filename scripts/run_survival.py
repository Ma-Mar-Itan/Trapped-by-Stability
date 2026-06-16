"""Stage 4 -- survival analysis in the HR+/HER2+ subgroup.

Requires the clustering stage to have run first.

Usage:
    python scripts/run_survival.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from radiomics_stability.statistics import survival


def main() -> None:
    survival.main()


if __name__ == "__main__":
    main()
