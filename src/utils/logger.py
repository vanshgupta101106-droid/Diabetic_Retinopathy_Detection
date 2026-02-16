"""
Centralized Logging Configuration
==================================

Provides a ``get_logger`` factory so every module shares the same
format and level without duplicating ``logging.basicConfig`` calls.
"""

import logging
import sys

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LEVEL = logging.INFO

# Configure root handler once (idempotent thanks to the guard)
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(_LEVEL)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a project logger with a consistent format.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured logging.Logger.
    """
    _configure_root()
    return logging.getLogger(name)
