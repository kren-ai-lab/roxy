"""Sequence-focused CLI package for Roxy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .main import app


def __getattr__(name: str) -> Any:
    """Resolve CLI exports lazily."""
    if name == "app":
        from .main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
