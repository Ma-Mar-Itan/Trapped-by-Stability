"""Stage 1 -- preprocessing: merge raw tables, build the radiomic feature matrix.

Usage:
    python scripts/run_preprocessing.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sets up sys.path)

from radiomics_stability.preprocessing import merge_data, prepare_radiomics


def main() -> None:
    merge_data.main()
    prepare_radiomics.main()


if __name__ == "__main__":
    main()
