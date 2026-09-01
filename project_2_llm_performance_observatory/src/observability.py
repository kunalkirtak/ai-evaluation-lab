"""
Optional observability integration (LangSmith).

LangSmith is NEVER required to run the benchmark. If the relevant
environment variables are not set, every function here becomes a no-op.
If LangSmith is enabled but the SDK is missing or a call fails, the
error is caught and logged; it never interrupts the benchmark run.
"""

import logging
from contextlib import contextmanager

from src import config

logger = logging.getLogger("llm_observatory.observability")

_client = None
_available = False

if config.ENABLE_LANGSMITH and config.LANGSMITH_TRACING:
    try:
        from langsmith import Client  # type: ignore

        _client = Client()
        _available = True
        logger.info("LangSmith tracing enabled")
    except Exception as exc:
        logger.warning(f"LangSmith requested but unavailable ({exc}); continuing without tracing")
        _client = None
        _available = False
else:
    logger.info("LangSmith tracing disabled (optional observability not configured)")


def is_enabled() -> bool:
    return _available


@contextmanager
def trace_run(run_name: str, run_type: str = "chain", **metadata):
    """
    Best-effort tracing context manager. Always yields, even if tracing
    is disabled or fails to start - the benchmark body always executes.
    """
    if not _available or _client is None:
        yield None
        return

    try:
        logger.info(f"LangSmith trace started | run={run_name}")
        yield _client
    except Exception as exc:
        logger.warning(f"LangSmith tracing error (ignored): {exc}")
        yield None
    finally:
        if _available:
            logger.info(f"LangSmith trace finished | run={run_name}")


def log_event(message: str, **fields) -> None:
    """Best-effort structured log event, safe to call even if disabled."""
    if not _available:
        return
    try:
        logger.info(f"[langsmith] {message} | {fields}")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"LangSmith log_event failed (ignored): {exc}")
