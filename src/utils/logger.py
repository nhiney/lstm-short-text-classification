"""
logger.py — Centralised logging setup.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to stdout and, optionally, a file.

    Args:
        name     : module name, typically ``__name__``
        log_file : path to a .log file; if None only stdout is used
        level    : logging level (default INFO)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
