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


class DescriptorBlockContractError(DescriptorError):
    """Errors raised when a descriptor block violates the declared contract."""


class SequenceError(DescriptorError):
    """Base class for sequence preprocessing and validation errors."""


class SequenceInputError(SequenceError):
    """Errors caused by unsupported or malformed sequence inputs."""


class SequenceCollectionError(SequenceInputError):
    """Errors caused by invalid sequence collection inputs or shape issues."""


class MissingSequenceError(SequenceInputError):
    """Errors raised when a sequence value is missing."""


class InvalidSequenceError(SequenceError):
    """Errors caused by invalid sequence content under strict rules."""


class EmptySequenceError(InvalidSequenceError):
    """Errors raised when a sequence is empty after normalization or cleaning."""


class SequenceTooShortError(InvalidSequenceError):
    """Errors raised when a sequence is shorter than the required length."""


class AAIndexError(DescriptorError):
    """Errors specific to AAIndex handling and descriptor computation."""


class ProjectionError(RoxyError):
    """Errors raised during dimensionality reduction or projection."""


class EDAError(RoxyError):
    """Errors related to exploratory data analysis or statistical tests."""
