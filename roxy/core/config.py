"""Roxy cache configuration and runtime overrides."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_LOCK = RLock()
_CACHE_ROOT: Path | None = None

_DEFAULT_CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "roxy"


def get_cache_root() -> Path:
    """Return current cache root (default: ``~/.cache/roxy``)."""
    with _LOCK:
        root = _CACHE_ROOT if _CACHE_ROOT is not None else _DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_cache_root(new_root: Path | str) -> None:
    """Override cache root programmatically."""
    global _CACHE_ROOT  # noqa: PLW0603,RUF100
    resolved = Path(new_root).expanduser().resolve()
    with _LOCK:
        _CACHE_ROOT = resolved
    resolved.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_cache_root(temp_root: Path | str) -> Generator[None, None, None]:
    """Context manager: temporarily override cache root."""
    prev = get_cache_root()
    set_cache_root(temp_root)
    try:
        yield
    finally:
        set_cache_root(prev)


__all__ = ["get_cache_root", "set_cache_root", "temporary_cache_root"]
