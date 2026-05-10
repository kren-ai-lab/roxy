"""Tests for the Roxy exception hierarchy."""

from roxy.core.exceptions import (
    AAIndexError,
    DescriptorError,
    RoxyError,
    RoxyIOError,
    SequenceValidationError,
)


def test_hierarchy():
    assert issubclass(DescriptorError, RoxyError)
    assert issubclass(AAIndexError, DescriptorError)
    assert issubclass(SequenceValidationError, DescriptorError)
    assert issubclass(RoxyIOError, RoxyError)


def test_raise_and_catch_base():
    try:
        raise AAIndexError("bad index")
    except RoxyError as exc:
        assert "bad index" in str(exc)
