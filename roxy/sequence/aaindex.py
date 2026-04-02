"""AAIndex-backed sequence descriptor helpers.

This module isolates the sequence-level AAIndex descriptor logic used by
the current sequence compatibility layer.

The current scope is intentionally small:
- normalize requested AAIndex codes,
- expose the current mean-based descriptor block,
- keep a clean bridge to the core AAIndex backend.

Edge-case policy
----------------
- Empty cleaned sequences return ``NaN`` for each requested AAIndex
  feature.
- If no AAIndex codes are requested, the descriptor block is empty.
- Unknown AAIndex codes return ``NaN`` per code via the core AAIndex
  backend instead of raising during descriptor computation.
- Unknown residues are handled by the calling cleaning policy. Direct
  low-level calls remain permissive and the backend averages only over
  canonical residues.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from roxy.core.exceptions import AAIndexError

try:
    from roxy.core.aaindex import (
        compute_aaindex_means_for_sequence as _compute_mean_descriptors,
        load_aaindex as _load_aaindex,
    )
except ImportError:  # pragma: no cover - standalone compatibility
    try:
        from core.aaindex import compute_aaindex_means_for_sequence as _compute_mean_descriptors  # type: ignore[assignment]
        from core.aaindex import load_aaindex as _load_aaindex  # type: ignore[assignment]
    except ImportError:  # pragma: no cover
        _compute_mean_descriptors = None  # type: ignore[assignment]
        _load_aaindex = None  # type: ignore[assignment]

from .utils import clean_sequence


def aaindex_backend_available() -> bool:
    """Return whether the AAIndex backend is importable."""
    return _compute_mean_descriptors is not None


def normalize_aaindex_codes(index_codes: Iterable[str]) -> List[str]:
    """Normalize AAIndex codes into a materialized, stripped list."""
    return [str(code).strip() for code in index_codes if str(code).strip()]


def empty_aaindex_mean_descriptors(index_codes: Iterable[str]) -> Dict[str, float]:
    """Return NaN-filled AAIndex mean descriptors for the requested codes."""
    return {
        f"aaindex_{code}_mean": float("nan")
        for code in normalize_aaindex_codes(index_codes)
    }


def validate_aaindex_codes(
    index_codes: Iterable[str],
    *,
    auto_download: bool = True,
) -> Dict[str, bool]:
    """Check whether requested AAIndex codes exist in the core AAIndex table.

    This helper is intentionally lightweight and non-strict so callers can
    inspect code availability without changing the current warning/NaN
    behavior used during descriptor computation.
    """
    codes = normalize_aaindex_codes(index_codes)
    if not codes:
        return {}
    if _load_aaindex is None:
        raise AAIndexError("AAIndex backend is not available.")

    table = _load_aaindex(auto_download=auto_download)
    return {code: code in table.columns for code in codes}


def compute_sequence_aaindex_means(
    seq: str,
    *,
    index_codes: Iterable[str],
) -> Dict[str, float]:
    """Compute per-sequence AAIndex mean descriptors.

    This preserves the current behavior by delegating to the core
    AAIndex backend, which handles table loading and unknown-code NaN
    fallbacks. Empty cleaned sequences yield ``NaN`` for each requested
    AAIndex code.
    """
    if _compute_mean_descriptors is None:
        msg = (
            "AAIndex support is not available. Make sure "
            "`roxy.core.aaindex` (or `core.aaindex` in standalone mode) "
            "is importable and AAIndex is configured."
        )
        raise AAIndexError(msg)

    codes = normalize_aaindex_codes(index_codes)
    if not codes:
        return {}

    return _compute_mean_descriptors(
        clean_sequence(seq),
        index_codes=codes,
    )


def aaindex_mean_descriptors(
    seq: str,
    *,
    index_codes: Iterable[str],
) -> Dict[str, float]:
    """Return the current AAIndex descriptor block for a sequence."""
    return compute_sequence_aaindex_means(
        seq,
        index_codes=index_codes,
    )


# TODO:
# - Add optional AAIndex summary variants such as std/min/max only after
#   they are introduced in the active descriptor contract.
# - Evaluate whether terminal or windowed AAIndex summaries should live in
#   this module or in dedicated descriptor-family modules later on.

__all__ = [
    "aaindex_backend_available",
    "aaindex_mean_descriptors",
    "compute_sequence_aaindex_means",
    "empty_aaindex_mean_descriptors",
    "normalize_aaindex_codes",
    "validate_aaindex_codes",
]
