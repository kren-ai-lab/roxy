"""Logging utilities for Roxy."""

from __future__ import annotations

import logging
from pathlib import Path

# Ensure the top-level 'roxy' logger has a NullHandler to avoid
# "No handler found" warnings when the host application has not
# configured logging.
_root_logger = logging.getLogger("roxy")
if not _root_logger.handlers:
    _root_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``roxy`` namespace.

    Args:
        name: Optional logger name. If ``None``, returns the top-level ``"roxy"``
            logger. Dotted names are attached as children, e.g. ``roxy.core.io``.

    """
    if name is None:
        return logging.getLogger("roxy")

    if name.startswith("roxy."):
        return logging.getLogger(name)

    return logging.getLogger(f"roxy.{name}")


def setup_logger(
    level: int = logging.INFO,
    fmt: str | None = None,
    log_file: str | Path | None = None,
    file_mode: str = "a",
) -> logging.Logger:
    """Configure a basic logger for Roxy.

    Optional convenience for quick experiments, notebooks and CLI use.
    Sets the log level on the ``"roxy"`` logger and attaches a
    ``StreamHandler`` (stderr) and optionally a ``FileHandler``.

    Args:
        level: Logging level, e.g. ``logging.INFO``.
        fmt: Log message format. Defaults to timestamp + level + logger name.
        log_file: Optional path to a log file.
        file_mode: File open mode — ``"a"`` (append) or ``"w"`` (overwrite).

    Returns:
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
