"""Shared utilities for composition descriptors."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

from roxy.core.constants import AA20

if TYPE_CHECKING:
    from collections.abc import Iterable

_AA20_SET: frozenset[str] = frozenset(AA20)


def clean_sequence(seq: str | None) -> str:
    """Strip whitespace, uppercase, remove stop codons, keep only standard AAs."""
    if not seq or not isinstance(seq, str):
        return ""
    return "".join(aa for aa in seq.strip().upper().replace("*", "") if aa in _AA20_SET)


def extract_kmers(seq: str, k: int) -> list[str]:
    """Return all overlapping k-mers from seq. Empty list if len(seq) < k."""
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def generate_all_kmers(alphabet: Iterable[str], k: int) -> list[str]:
    """Return sorted list of all k-mers over alphabet (cartesian product)."""
    return ["".join(p) for p in product(sorted(alphabet), repeat=k)]


def safe_ratio(a: float, b: float) -> float:
    """Return a/b, or NaN if b is zero."""
    return a / b if b != 0 else math.nan


def safe_log2(x: float) -> float:
    """Return log2(x) for x > 0, else 0.0."""
    return math.log2(x) if x > 0 else 0.0


__all__ = [
    "clean_sequence",
    "extract_kmers",
    "generate_all_kmers",
    "safe_log2",
    "safe_ratio",
]
