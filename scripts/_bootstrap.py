"""Make the ``src`` layout importable when running scripts directly.

Allows ``python scripts/run_full_pipeline.py`` to work from a fresh clone
without requiring ``pip install -e .`` first. Import this module before any
``radiomics_stability`` import.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
