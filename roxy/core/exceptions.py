"""Custom exception hierarchy for Roxy."""

from __future__ import annotations


class RoxyError(Exception):
    """Base class for all Roxy-specific errors."""


class DescriptorError(RoxyError):
    """Errors raised by descriptor engines or descriptor utilities."""


class SequenceValidationError(DescriptorError):
    """Invalid or unrecognised amino-acid sequence."""


class AAIndexError(DescriptorError):
    """Errors specific to AAIndex loading and computation."""


class RoxyIOError(RoxyError):
    """Errors related to reading/writing sequence or feature files."""


__all__ = [
    "AAIndexError",
    "DescriptorError",
    "RoxyError",
    "RoxyIOError",
    "SequenceValidationError",
]
