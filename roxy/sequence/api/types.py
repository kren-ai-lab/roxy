"""Internal type aliases for the sequence API package."""

from __future__ import annotations

from typing import Any, Tuple

SequenceInput = Any
FastaRecord = Tuple[str, str]

__all__ = [
    "FastaRecord",
    "SequenceInput",
]
