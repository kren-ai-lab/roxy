"""Roxy core package."""

from __future__ import annotations

from .config import get_cache_root, set_cache_root, temporary_cache_root
from .exceptions import (
    AAIndexError,
    DescriptorError,
    RoxyError,
    RoxyIOError,
    SequenceValidationError,
)
from .io import read_fasta, read_sequences, write_table

__all__ = [
    "AAIndexError",
    "DescriptorError",
    "RoxyError",
    "RoxyIOError",
    "SequenceValidationError",
    "get_cache_root",
    "read_fasta",
    "read_sequences",
    "set_cache_root",
    "temporary_cache_root",
    "write_table",
]
