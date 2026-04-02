"""Shared utilities for sequence descriptor computation.

This module hosts low-risk scalar helpers reused across descriptor
families. Canonical preprocessing policy now lives in
``roxy.sequence.cleaning`` and ``roxy.sequence.validation``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Union

from .cleaning import (
    normalize_sequence,
    remove_terminal_stop_marker,
)


def clean_sequence(seq: str) -> str:
    """Normalize a sequence string for descriptor computation.

    This low-level helper preserves the current descriptor-module
    behavior by:
    - uppercasing and trimming external whitespace,
    - keeping non-canonical symbols unchanged,
    - removing stop markers so direct module calls stay permissive.

    The public sequence API performs stricter validation before calling
    descriptor blocks.
    """
    normalized = normalize_sequence(
        seq,
        uppercase=True,
        strip_external_whitespace=True,
        remove_internal_whitespace=False,
    )
    if normalized is None:
        return ""
    normalized = remove_terminal_stop_marker(normalized)
    return normalized.replace("*", "")


def mean_scale(seq: str, scale: Dict[str, float]) -> float:
    """Return the mean value of a residue-level scale over a sequence."""
    values = [scale[aa] for aa in clean_sequence(seq) if aa in scale]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def count_residues(seq: str, residues: Iterable[str]) -> int:
    """Count the total number of residues from a set within a sequence."""
    s = clean_sequence(seq)
    return int(sum(s.count(residue) for residue in residues))


def sum_residue_values(
    seq: str,
    values: Mapping[str, Union[int, float]],
) -> float:
    """Sum per-residue values over a cleaned sequence."""
    s = clean_sequence(seq)
    return float(sum(values.get(residue, 0) for residue in s))


def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    *,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning a default for zero denominator."""
    if denominator == 0:
        return float(default)
    return float(numerator / denominator)


def shannon_entropy(seq: str, alphabet: Iterable[str]) -> float:
    """Return Shannon entropy in bits over the provided alphabet."""
    s = clean_sequence(seq)
    length = len(s)
    if length == 0:
        return math.nan

    probs: List[float] = []
    for token in alphabet:
        count = s.count(token)
        if count:
            probs.append(count / length)

    if not probs:
        return math.nan
    return float(-sum(p * math.log2(p) for p in probs))


def linguistic_complexity(
    seq: str,
    *,
    max_k: int = 3,
    alphabet_size: int = 20,
) -> Dict[str, float]:
    """Compute simple k-mer linguistic complexity for `k = 1..max_k`."""
    s = clean_sequence(seq)
    length = len(s)
    if length == 0:
        return {f"lc_k{k}": 0.0 for k in range(1, max_k + 1)}

    feats: Dict[str, float] = {}
    for k in range(1, max_k + 1):
        if length < k:
            feats[f"lc_k{k}"] = 0.0
            continue
        observed = {s[i : i + k] for i in range(length - k + 1)}
        max_possible = min(alphabet_size**k, length - k + 1)
        feats[f"lc_k{k}"] = (
            len(observed) / max_possible if max_possible > 0 else 0.0
        )
    return feats


__all__ = [
    "clean_sequence",
    "count_residues",
    "safe_divide",
    "mean_scale",
    "shannon_entropy",
    "sum_residue_values",
    "linguistic_complexity",
]
