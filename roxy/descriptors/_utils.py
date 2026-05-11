"""Shared utilities for descriptor implementations."""

from __future__ import annotations

import math
from collections import Counter
from itertools import groupby, product
from typing import TYPE_CHECKING

import numpy as np

from roxy.core.constants import AA20, PKA_C_TERM, PKA_N_TERM, PKA_SIDE

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Set as AbstractSet

_AA20_SET: frozenset[str] = frozenset(AA20)
_NAN = math.nan
_POSITIVE_IONIZABLE = frozenset("KRH")
_NEGATIVE_IONIZABLE = frozenset("DECY")


def clean_sequence(seq: str | None) -> str:
    """Strip whitespace, uppercase, remove stop codons, keep only standard AAs."""
    if not seq or not isinstance(seq, str):
        return ""
    return "".join(aa for aa in seq.strip().upper().replace("*", "") if aa in _AA20_SET)


def generate_all_kmers(alphabet: Iterable[str], k: int) -> list[str]:
    """Return sorted list of all k-mers over alphabet."""
    return ["".join(p) for p in product(sorted(alphabet), repeat=k)]


def safe_ratio(a: float, b: float) -> float:
    """Return a/b, or NaN if b is zero."""
    return a / b if b != 0 else _NAN


def windows(seq: str, size: int) -> list[str]:
    """Return all overlapping substrings of length ``size``."""
    if len(seq) < size:
        return []
    return [seq[i : i + size] for i in range(len(seq) - size + 1)]


def scale_values(seq: str, scale: dict[str, float]) -> list[float]:
    """Map each residue in seq through scale; silently skip missing AAs."""
    return [scale[aa] for aa in seq if aa in scale]


def scale_mean(seq: str, scale: dict[str, float]) -> float:
    """Mean of scale values over seq; NaN if seq is empty."""
    vals = scale_values(seq, scale)
    return float(np.mean(vals)) if vals else _NAN


def scale_std(seq: str, scale: dict[str, float]) -> float:
    """Return population std of scale values over seq; NaN if seq is empty."""
    vals = scale_values(seq, scale)
    return float(np.std(vals, ddof=0)) if vals else _NAN


def zscore_scale(scale: dict[str, float], alphabet: Iterable[str] = AA20) -> dict[str, float]:
    """Return z-score normalized values for a residue scale."""
    residues = sorted(alphabet)
    vals = np.array([scale[aa] for aa in residues], dtype=float)
    mean, std = vals.mean(), vals.std(ddof=0)
    if std == 0:
        return dict.fromkeys(residues, 0.0)
    return {aa: (scale[aa] - mean) / std for aa in residues}


def fraction_from_group(seq: str, group: AbstractSet[str]) -> float:
    """Fraction of seq residues that belong to group; NaN if empty."""
    n = len(seq)
    if n == 0:
        return _NAN
    return sum(aa in group for aa in seq) / n


def profile_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics over a profile vector."""
    if not values:
        return {
            "mean": _NAN,
            "std": _NAN,
            "min": _NAN,
            "max": _NAN,
            "amplitude": _NAN,
            "start_end_diff": _NAN,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "amplitude": float(arr.max() - arr.min()),
        "start_end_diff": float(arr[-1] - arr[0]),
    }


def fraction_above_threshold(values: list[float], threshold: float) -> float:
    """Fraction of values strictly above threshold; NaN if empty."""
    if not values:
        return _NAN
    return float(np.mean(np.array(values) > threshold))


def terminal_segment(seq: str, side: str, window: int) -> str:
    """Return N- or C-terminal segment of length up to window."""
    return seq[:window] if side == "N" else seq[-window:]


def longest_run(seq: str, group: AbstractSet[str]) -> int:
    """Length of the longest consecutive run of residues in group."""
    if not seq:
        return 0
    best = 0
    cur = 0
    for aa in seq:
        if aa in group:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def transition_fraction(
    seq: str,
    group_a: AbstractSet[str],
    group_b: AbstractSet[str],
) -> float:
    """Fraction of adjacent pairs that switch between group_a and group_b."""
    if len(seq) < 2:  # noqa: PLR2004
        return _NAN
    transitions = sum(
        (seq[i] in group_a) != (seq[i + 1] in group_a)
        and (seq[i] in group_a or seq[i] in group_b)
        and (seq[i + 1] in group_a or seq[i + 1] in group_b)
        for i in range(len(seq) - 1)
    )
    return transitions / (len(seq) - 1)


def net_charge_at_ph(seq: str, ph: float) -> float:
    """Compute Henderson-Hasselbalch net charge including termini."""
    if not seq:
        return _NAN
    pos = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))
    neg = 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in _POSITIVE_IONIZABLE:
            pos += 1.0 / (1.0 + 10 ** (ph - pka))
        elif aa in _NEGATIVE_IONIZABLE:
            neg += 1.0 / (1.0 + 10 ** (pka - ph))
    return pos - neg


def longest_homopolymer_run(seq: str) -> int:
    """Length of the longest run of identical residues."""
    if not seq:
        return 0
    return max(len(list(g)) for _, g in groupby(seq))


def linguistic_complexity(seq: str, k: int) -> float:
    """Fraction of possible k-mers observed in seq; NaN if seq is too short."""
    if len(seq) < k or k < 1:
        return _NAN
    observed = len(set(windows(seq, k)))
    possible = min(len(seq) - k + 1, 20**k)
    return observed / possible if possible > 0 else _NAN


def shannon_entropy(seq: str) -> float:
    """Return Shannon entropy in bits over symbols in seq."""
    if not seq:
        return _NAN
    counts = Counter(seq)
    total = len(seq)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


__all__ = [
    "clean_sequence",
    "fraction_above_threshold",
    "fraction_from_group",
    "generate_all_kmers",
    "linguistic_complexity",
    "longest_homopolymer_run",
    "longest_run",
    "net_charge_at_ph",
    "profile_stats",
    "safe_ratio",
    "scale_mean",
    "scale_std",
    "scale_values",
    "shannon_entropy",
    "terminal_segment",
    "transition_fraction",
    "windows",
    "zscore_scale",
]
