"""Roxy core package.

This subpackage contains low-level building blocks shared across the
sequence-focused Roxy package:

- :mod:`roxy.core.aaindex` - AAIndex cache and lookup utilities.
- :mod:`roxy.core.constants` - amino-acid scales and related constants.
- :mod:`roxy.core.exceptions` - custom exception hierarchy.
- :mod:`roxy.core.logging_utils` - logging helpers.
"""

from __future__ import annotations

from roxy.core.exceptions import (
    RoxyError,
    DescriptorError,
    AAIndexError,
    EmptySequenceError,
    InvalidSequenceError,
    SequenceCollectionError,
    SequenceError,
    SequenceInputError,
)
from roxy.core.logging_utils import get_logger, setup_logger

logger = get_logger(__name__)


def ensure_aaindex_available() -> None:
    """Ensure that the AAIndex backend is available.

    The import is performed lazily so importing `roxy.core` does not
    force AAIndex backend dependencies unless AAIndex functionality is
    explicitly requested.
    """
    from roxy.core.aaindex import ensure_aaindex_available as _ensure

    _ensure()

# NOTE:
# - Avoid AAIndex cache initialization at import time to keep package
#   imports side-effect free.
# - Call `ensure_aaindex_available()` explicitly from sequence-facing APIs
#   when AAIndex-backed descriptors are requested.

__all__ = [
    "RoxyError",
    "DescriptorError",
    "AAIndexError",
    "SequenceError",
    "SequenceInputError",
    "SequenceCollectionError",
    "InvalidSequenceError",
    "EmptySequenceError",
    "get_logger",
    "setup_logger",
    "ensure_aaindex_available",
]
