"""Single logging configuration used across the whole pipeline.

Replaces the scattered ``print`` calls of the original scripts with a
consistent, timestamped logger. Call :func:`get_logger` at module import.
"""

from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a stream handler on the root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (typically ``__name__``)."""
    configure_logging()
    return logging.getLogger(name)
