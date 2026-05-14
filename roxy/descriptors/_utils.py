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
    """Strip whitespace, uppercase, remove stop codons, and keep only standard AAs."""
    if not seq or not isinstance(seq, str):
        return ""
    return "".join(aa for aa in seq.strip().upper().replace("*", "") if aa in _AA20_SET)


def generate_all_kmers(alphabet: Iterable[str], k: int) -> list[str]:
    """Return sorted list of all k-mers over ``alphabet``."""
    return ["".join(p) for p in product(sorted(alphabet), repeat=k)]


def safe_ratio(a: float, b: float) -> float:
    """Return ``a / b``, or NaN if ``b`` is zero."""
    return a / b if b != 0 else _NAN


def windows(seq: str, size: int) -> list[str]:
    """Return all overlapping substrings of length ``size``.

    Yields ``len(seq) - size + 1`` windows, or an empty list when
    ``len(seq) < size``.
    """
    if len(seq) < size:
        return []
    return [seq[i : i + size] for i in range(len(seq) - size + 1)]


def scale_values(seq: str, scale: dict[str, float]) -> list[float]:
    """Map each residue in ``seq`` through ``scale``, skipping missing AAs."""
    return [scale[aa] for aa in seq if aa in scale]


def scale_array(seq: str, scale: dict[str, float]) -> np.ndarray:
    """Map each residue in ``seq`` through ``scale``; missing values become NaN."""
    return np.array([scale.get(aa, np.nan) for aa in seq], dtype=float)


def membership_array(seq: str, group: AbstractSet[str]) -> np.ndarray:
    """Return a 0/1 array indicating whether each residue is in ``group``."""
    return np.fromiter((aa in group for aa in seq), dtype=float, count=len(seq))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return valid-window rolling means for a numeric vector.

    For a vector of length *N*, produces ``N - window + 1`` output values.
    Returns an empty array when ``window < 1`` or ``N < window``.
    """
    if window < 1 or values.size < window:
        return np.array([], dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def scale_mean(seq: str, scale: dict[str, float]) -> float:
    """Mean of ``scale`` values over ``seq``; NaN if ``seq`` is empty."""
    vals = scale_values(seq, scale)
    return float(np.mean(vals)) if vals else _NAN


def scale_std(seq: str, scale: dict[str, float]) -> float:
    """Compute population standard deviation of ``scale`` values over ``seq``.

    Returns NaN if ``seq`` is empty.
    """
    vals = scale_values(seq, scale)
    return float(np.std(vals, ddof=0)) if vals else _NAN


def zscore_scale(scale: dict[str, float], alphabet: Iterable[str] = AA20) -> dict[str, float]:
    r"""Return z-score normalized values for a residue scale.

    Each value is transformed as:

    .. math::

        z = \frac{x - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are the population mean and
    standard deviation over ``alphabet``.
    """
    residues = sorted(alphabet)
    vals = np.array([scale[aa] for aa in residues], dtype=float)
    mean, std = vals.mean(), vals.std(ddof=0)
    if std == 0:
        return dict.fromkeys(residues, 0.0)
    return {aa: (scale[aa] - mean) / std for aa in residues}


def fraction_from_group(seq: str, group: AbstractSet[str]) -> float:
    """Fraction of ``seq`` residues that belong to ``group``; NaN if empty."""
    n = len(seq)
    if n == 0:
        return _NAN
    return sum(aa in group for aa in seq) / n


def profile_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    """Compute summary statistics (mean, std, min, max, amplitude, start_end_diff) over a profile vector."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": _NAN,
            "std": _NAN,
            "min": _NAN,
            "max": _NAN,
            "amplitude": _NAN,
            "start_end_diff": _NAN,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "amplitude": float(arr.max() - arr.min()),
        "start_end_diff": float(arr[-1] - arr[0]),
    }


def fraction_above_threshold(values: list[float] | np.ndarray, threshold: float) -> float:
    """Fraction of ``values`` strictly above ``threshold``; NaN if empty."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return _NAN
    return float(np.mean(arr > threshold))


def terminal_segment(seq: str, side: str, window: int) -> str:
    """Return N- or C-terminal segment of length up to ``window``."""
    return seq[:window] if side == "N" else seq[-window:]


def longest_run(seq: str, group: AbstractSet[str]) -> int:
    """Length of the longest consecutive run of residues in ``group``."""
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
    """Fraction of adjacent pairs that switch between ``group_a`` and ``group_b``."""
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
    r"""Compute net charge at a given pH via the Henderson-Hasselbalch equation.

    Positive contributions (N-terminus and K, R, H side chains):

    .. math::

        q^{+} = \frac{1}{1 + 10^{(\text{pH} - \text{p}K_a)}}

    Negative contributions (C-terminus and D, E, C, Y side chains):

    .. math::

        q^{-} = \frac{1}{1 + 10^{(\text{p}K_a - \text{pH})}}

    Returns :math:`\sum q^{+} - \sum q^{-}`, or NaN if ``seq`` is empty.
    """
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
    """Length of the longest run of identical consecutive residues."""
    if not seq:
        return 0
    return max(len(list(g)) for _, g in groupby(seq))


def linguistic_complexity(seq: str, k: int) -> float:
    r"""Fraction of possible k-mers observed in ``seq``.

    .. math::

        LC(k) = \frac{|\text{observed}|}{\min(N - k + 1,\; 20^k)}

    where *N* is the sequence length. Returns NaN if ``N < k`` or ``k < 1``.
    """
    if len(seq) < k or k < 1:
        return _NAN
    observed = len(set(windows(seq, k)))
    possible = min(len(seq) - k + 1, 20**k)
    return observed / possible if possible > 0 else _NAN


def shannon_entropy(seq: str) -> float:
    r"""Shannon entropy (bits) over symbol frequencies in ``seq``.

    .. math::

        H = -\sum_{i} p_i \log_2(p_i)

    where :math:`p_i` is the relative frequency of each distinct symbol.
    Returns NaN if ``seq`` is empty.
    """
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
    "membership_array",
    "net_charge_at_ph",
    "profile_stats",
    "rolling_mean",
    "safe_ratio",
    "scale_array",
    "scale_mean",
    "scale_std",
    "scale_values",
    "shannon_entropy",
    "terminal_segment",
    "transition_fraction",
    "windows",
    "zscore_scale",
]
