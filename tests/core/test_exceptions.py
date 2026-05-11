"""Tests for the remaining Roxy exception hierarchy."""

from roxy.core.exceptions import (
    RoxyError,
    RoxyIOError,
)


def test_hierarchy():
    assert issubclass(RoxyIOError, RoxyError)


def test_raise_and_catch_base():
    try:
        raise RoxyIOError("bad io")
    except RoxyError as exc:
        assert "bad io" in str(exc)
