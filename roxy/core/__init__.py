"""Roxy core package.

This subpackage contains low-level building blocks that are shared across
the rest of the library, including:

- :mod:`roxy.core.dataset` – the :class:`RoxyDataset` container.
- :mod:`roxy.core.aaindex` – AAIndex cache and lookup utilities.
- :mod:`roxy.core.constants` – amino-acid scales and related constants.
- :mod:`roxy.core.report` – lightweight report dataclasses.
- :mod:`roxy.core.exceptions` – custom exception hierarchy.
- :mod:`roxy.core.logging_utils` – logging helpers.
"""

from __future__ import annotations

from roxy.core.aaindex import ensure_aaindex_available
from roxy.core.exceptions import (
    RoxyError,
    DatasetError,
    DescriptorError,
    AAIndexError,
    ProjectionError,
    EDAError,
)
from roxy.core.logging_utils import get_logger, setup_logger

logger = get_logger(__name__)

# Try to warm up AAIndex, but do not fail hard at import time.
# This may trigger a download on first import if the cache is empty.
try:  # pragma: no cover (import-time side effect)
    ensure_aaindex_available()
except AAIndexError as exc:  # pragma: no cover
    logger.warning(
        "AAIndex could not be initialised at import time: %s. "
        "AAIndex-based sequence descriptors may be unavailable until "
        "the cache is populated.",
        exc,
    )

__all__ = [
    "RoxyError",
    "DatasetError",
    "DescriptorError",
    "AAIndexError",
    "ProjectionError",
    "EDAError",
    "get_logger",
    "setup_logger",
    "ensure_aaindex_available",
]