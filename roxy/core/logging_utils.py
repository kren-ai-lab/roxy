from __future__ import annotations

"""Logging utilities for Roxy.

This module provides a small helper to obtain namespaced loggers for the
library, as well as an optional convenience function to configure
console/file handlers. By default, Roxy does **not** configure global
logging; it only attaches a :class:`logging.NullHandler` to the top-level
``"roxy"`` logger so that importing the library does not emit warnings
if the application has not configured logging.

Typical usage within the library::

    from roxy.core.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("Starting descriptor computation for %d samples", n_samples)

For applications, CLI or notebooks, you can call
:func:`setup_logger` once to configure handlers::

    from roxy.core.logging_utils import setup_logger
    setup_logger(level=logging.INFO, log_file="logs/roxy.log")
"""

import logging
from pathlib import Path
from typing import Optional, Union


# Ensure the top-level 'roxy' logger has a NullHandler to avoid
# "No handler found" warnings when the host application has not
# configured logging.
_root_logger = logging.getLogger("roxy")
if not _root_logger.handlers:
    _root_logger.addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the ``roxy`` namespace.

    Parameters
    ----------
    name:
        Optional fully-qualified logger name. If ``None``, the top-level
        ``"roxy"`` logger is returned. When a dotted name is provided,
        it is attached as a child of ``"roxy"``, e.g. ``roxy.core.dataset``.
    """
    if name is None:
        return logging.getLogger("roxy")

    if name.startswith("roxy."):
        return logging.getLogger(name)

    return logging.getLogger(f"roxy.{name}")


def setup_logger(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    file_mode: str = "a",
) -> logging.Logger:
    """Configure a basic logger for Roxy.

    This is an optional convenience for quick experiments, notebooks and
    CLI use. It sets the log level on the ``"roxy"`` logger and attaches:

    - a :class:`logging.StreamHandler` to stderr (console), and
    - optionally a :class:`logging.FileHandler` if ``log_file`` is provided.

    Parameters
    ----------
    level:
        Logging level to use, e.g. :data:`logging.INFO`.
    fmt:
        Optional log message format. If omitted, a default including
        timestamp, level and logger name is used.
    log_file:
        Optional path to a log file. If provided, a file handler is
        attached writing to this file.
    file_mode:
        File mode for the log file, e.g. ``"a"`` (append, default) or
        ``"w"`` (overwrite).

    Returns
    -------
    logging.Logger
        The configured top-level ``"roxy"`` logger.
    """
    logger = logging.getLogger("roxy")
    logger.setLevel(level)

    if fmt is None:
        fmt = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"

    formatter = logging.Formatter(fmt)

    # Console handler: attach only if none exists yet
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path, mode=file_mode, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
