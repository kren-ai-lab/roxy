"""Custom exception hierarchy for Roxy."""

from __future__ import annotations


class RoxyError(Exception):
    """Base class for all Roxy-specific errors."""


class RoxyIOError(RoxyError):
    """Errors related to reading/writing sequence or feature files."""


__all__ = [
    "RoxyError",
    "RoxyIOError",
]
