"""Runtime helpers for opt-in core initialization tasks."""

from __future__ import annotations


def ensure_aaindex_available() -> None:
    """Ensure that the AAIndex backend is available.

    The import is performed lazily so importing ``roxy.core`` does not
    force AAIndex backend dependencies unless AAIndex functionality is
    explicitly requested.
    """
    from roxy.core.aaindex import ensure_aaindex_available as _ensure

    _ensure()


__all__ = ["ensure_aaindex_available"]
