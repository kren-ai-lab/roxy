"""Custom exception hierarchy for Roxy."""

from __future__ import annotations


class RoxyError(Exception):
    """Base class for all Roxy-specific errors."""


class RoxyIOError(RoxyError):
    """Errors related to reading/writing sequence or feature files."""


class RoxyIdError(RoxyIOError):
    """Empty or duplicated record ids (bad input data)."""


class RoxyIdColumnError(RoxyIdError):
    """The id column is missing from a table (reported as a usage error)."""


__all__ = [
    "RoxyError",
    "RoxyIOError",
    "RoxyIdColumnError",
    "RoxyIdError",
]
