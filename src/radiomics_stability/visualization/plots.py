"""Shared, low-level plotting helpers.

These factor out the scatter-by-group pattern that the original scripts repeated
across the manufacturer-overlap and residualization-comparison figures.
Stage-specific figure composition stays in the relevant analysis modules; only
the reusable primitives live here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def scatter_by_group(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    groups: Iterable,
    *,
    sort_groups: bool = False,
    label_fmt: str = "{}",
    alpha: float = 0.7,
    size: float = 35,
) -> None:
    """Scatter ``(x, y)`` colored by ``groups``, one matplotlib series per group.

    NaN group values are skipped. With ``sort_groups`` the unique group values
    are plotted in sorted order (used for numeric cluster labels).
    """
    plot_df = pd.DataFrame({"x": x, "y": y, "g": list(groups)})
    uniques = plot_df["g"].dropna().unique()
    if sort_groups:
        uniques = sorted(uniques)
    for value in uniques:
        sub = plot_df[plot_df["g"] == value]
        ax.scatter(sub["x"], sub["y"], label=label_fmt.format(value), alpha=alpha, s=size)
    ax.grid(True, alpha=0.3)
