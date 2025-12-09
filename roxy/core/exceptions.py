from __future__ import annotations

"""Custom exception hierarchy for Roxy.

These exception classes provide a small but expressive hierarchy for
errors raised within the Roxy library. Using dedicated exception types
instead of bare ``ValueError`` or ``RuntimeError`` makes it easier for
callers to catch and handle specific failure modes.
"""


class RoxyError(Exception):
    """Base class for all Roxy-specific errors."""


class DatasetError(RoxyError):
    """Errors related to dataset construction, validation or alignment."""


class DescriptorError(RoxyError):
    """Errors raised by descriptor engines or descriptor utilities."""


class AAIndexError(DescriptorError):
    """Errors specific to AAIndex handling and descriptor computation."""


class ProjectionError(RoxyError):
    """Errors raised during dimensionality reduction or projection."""


class EDAError(RoxyError):
    """Errors related to exploratory data analysis or statistical tests."""
