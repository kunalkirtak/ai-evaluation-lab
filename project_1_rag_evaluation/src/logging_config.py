"""Centralized logging setup for the RAG evaluation project."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return a module logger.

    Safe to call multiple times; only configures handlers on first call.
    """
    global _CONFIGURED
    logger = logging.getLogger("rag_eval")

    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    _CONFIGURED = True
    return logger
