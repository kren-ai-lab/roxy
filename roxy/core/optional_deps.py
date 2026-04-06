"""Helpers for optional third-party dependencies.

This module centralizes lazy imports for dependencies that some Roxy
surfaces may use opportunistically. Keeping the import guards here keeps
business modules focused on domain logic and user-facing behavior.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def _missing_dependency_message(
    package_name: str,
    *,
    purpose: str,
    install_hint: str,
) -> str:
    """Build a user-oriented optional-dependency error message."""
    return (
        f"{package_name} is required for {purpose}. "
        f"{install_hint}"
    )


def require_pandas(*, purpose: str) -> Any:
    """Import pandas for a table-oriented feature or raise a clear error."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            _missing_dependency_message(
                "pandas",
                purpose=purpose,
                install_hint="Install `pandas` to use this feature.",
            )
        ) from exc
    return pd


def require_numpy(*, purpose: str) -> Any:
    """Import NumPy for a numeric feature or raise a clear error."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            _missing_dependency_message(
                "numpy",
                purpose=purpose,
                install_hint="Install `numpy` to use this feature.",
            )
        ) from exc
    return np


def require_matplotlib(*, purpose: str) -> tuple[Any, Any, Any]:
    """Import Matplotlib plotting helpers or raise a clear error."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            _missing_dependency_message(
                "matplotlib",
                purpose=purpose,
                install_hint="Install `matplotlib` to use this feature.",
            )
        ) from exc
    return plt, Axes, Figure


def is_missing_optional_dependency(
    exc: BaseException,
    package_names: Collection[str],
) -> bool:
    """Return whether an import failure was caused by a known optional dependency."""
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        missing_name = getattr(current, "name", None)
        if missing_name in package_names:
            return True

        message = str(current)
        if any(package_name in message for package_name in package_names):
            return True

        current = current.__cause__ or current.__context__

    return False


__all__ = [
    "is_missing_optional_dependency",
    "require_matplotlib",
    "require_numpy",
    "require_pandas",
]
