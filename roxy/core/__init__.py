"""Roxy core package.

This subpackage contains low-level building blocks shared across the
sequence-focused Roxy package:

- :mod:`roxy.core.aaindex` - AAIndex cache and lookup utilities.
- :mod:`roxy.core.constants` - amino-acid scales and related constants.
- :mod:`roxy.core.exceptions` - custom exception hierarchy.
- :mod:`roxy.core.logging_utils` - logging helpers.
"""

from .exceptions import (
    AAIndexError,
    DescriptorError,
    EmptySequenceError,
    InvalidSequenceError,
    RoxyError,
    SequenceCollectionError,
    SequenceError,
    SequenceInputError,
)
from .logging_utils import get_logger, setup_logger
from .runtime import ensure_aaindex_available

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
