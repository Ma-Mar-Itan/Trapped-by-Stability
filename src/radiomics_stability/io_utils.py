"""Typed CSV / figure I/O helpers with consistent logging.

These thin wrappers remove the repeated ``pd.read_csv`` / ``to_csv`` /
``savefig`` boilerplate from the original scripts and guarantee that output
directories exist before writing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd

from radiomics_stability.config import FIGURE_DPI
from radiomics_stability.logging_utils import get_logger

logger = get_logger(__name__)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV into a DataFrame, logging its shape."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run the upstream pipeline stage first (see scripts/)."
        )
    df = pd.read_csv(path, **kwargs)
    logger.info("Loaded %s  shape=%s", Path(path).name, df.shape)
    return df


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Saved   %s  shape=%s", path.name, df.shape)
    return path


def save_figure(fig: matplotlib.figure.Figure, path: Path, dpi: int = FIGURE_DPI) -> Path:
    """Save a matplotlib figure at the configured resolution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure %s", path.name)
    return path
