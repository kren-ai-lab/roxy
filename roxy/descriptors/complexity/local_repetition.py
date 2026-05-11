"""Local repetition and sequence redundancy descriptors."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import groupby

import numpy as np

from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan
_MIN_REPEAT = 2


def _kmers(seq: str, k: int) -> list[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def _repeated_fraction(words: list[str]) -> float:
    if not words:
        return _NAN
    counts = Counter(words)
    return sum(v for v in counts.values() if v > 1) / len(words)


def _unique_fraction(words: list[str]) -> float:
    if not words:
        return _NAN
    return len(set(words)) / len(words)


def _redundancy_score(words: list[str]) -> float:
    if not words:
        return _NAN
    return 1.0 - _unique_fraction(words)


def _top_count(words: list[str]) -> float:
    if not words:
        return _NAN
    return float(max(Counter(words).values()))


def _top_freq(words: list[str]) -> float:
    if not words:
        return _NAN
    return _top_count(words) / len(words)


def _dup_window_frac(seq: str, window: int) -> float:
    words = _kmers(seq, window)
    if not words:
        return _NAN
    counts = Counter(words)
    return sum(1 for w in words if counts[w] > 1) / len(words)


def _repeated_burden(seq: str, k: int) -> float:
    words = _kmers(seq, k)
    if not words:
        return _NAN
    counts = Counter(words)
    return sum(v for v in counts.values() if v > 1) / len(words)


def _recurrence_entropy(words: list[str]) -> float:
    if not words:
        return _NAN
    counts = Counter(words)
    reps = np.array([v for v in counts.values() if v > 1], dtype=float)
    if len(reps) == 0:
        return 0.0
    probs = reps / reps.sum()
    return float(-(probs * np.log2(probs)).sum())


def _repeated_span(seq: str, k: int) -> float:
    words = _kmers(seq, k)
    if not words:
        return _NAN
    positions: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(words):
        positions[w].append(i)
    spans = [pos[-1] - pos[0] for pos in positions.values() if len(pos) >= _MIN_REPEAT]
    if not spans:
        return _NAN
    return float(np.mean(spans))


def _repeated_block_density(seq: str, k: int) -> float:
    words = _kmers(seq, k)
    if not words:
        return _NAN
    counts = Counter(words)
    return sum(1 for w in words if counts[w] > 1) / len(words)


def _longest_redundant_block(seq: str, k: int) -> float:
    words = _kmers(seq, k)
    if not words:
        return _NAN
    counts = Counter(words)
    binary = [1 if counts[w] > 1 else 0 for w in words]
    runs = [len(list(g)) for val, g in groupby(binary) if val == 1]
    return float(max(runs)) if runs else 0.0


def _recurrence_concentration(words: list[str]) -> float:
    if not words:
        return _NAN
    counts = Counter(words)
    repeated = [v for v in counts.values() if v > 1]
    if not repeated:
        return 0.0
    return max(repeated) / sum(repeated)


def _local_redundancy_profile(seq: str, outer_window: int, inner_k: int) -> list[float]:
    if len(seq) < outer_window or outer_window < inner_k:
        return []
    return [
        _redundancy_score(_kmers(seq[i : i + outer_window], inner_k))
        for i in range(len(seq) - outer_window + 1)
    ]


def _local_redundancy_mean(seq: str, outer_window: int, inner_k: int) -> float:
    profile = _local_redundancy_profile(seq, outer_window, inner_k)
    if not profile:
        return _NAN
    return float(np.mean(profile))


def _local_redundancy_std(seq: str, outer_window: int, inner_k: int) -> float:
    profile = _local_redundancy_profile(seq, outer_window, inner_k)
    if not profile:
        return _NAN
    return float(np.std(profile, ddof=0))


@register("local_repetition", family="complexity")
class LocalRepetitionDescriptor(BaseDescriptor):
    """Local repetition and k-mer redundancy descriptors.

    Captures sequence-level repetitiveness using k-mer statistics (repeated
    fraction, unique fraction, redundancy score, top word counts, recurrence
    entropy/concentration, span, block density) for k=2,3,4, duplicate-window
    fractions (w=5,6), and local redundancy profiles (outer_window=8,10).

    Output columns (prefix ``local_repetition_``):
        ``length``, ``valid_residue_count``, 33 feature columns.
        Total: 35 columns.
    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute local repetition features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        w2 = _kmers(seq, 2)
        w3 = _kmers(seq, 3)
        w4 = _kmers(seq, 4)

        def _span_norm(seq: str, k: int) -> float:
            span = _repeated_span(seq, k)
            return span / n if n > 0 and not math.isnan(span) else _NAN

        return {
            "length": float(n),
            "valid_residue_count": float(n),
            "repeated_fraction_k2": _repeated_fraction(w2),
            "repeated_fraction_k3": _repeated_fraction(w3),
            "repeated_fraction_k4": _repeated_fraction(w4),
            "unique_fraction_k2": _unique_fraction(w2),
            "unique_fraction_k3": _unique_fraction(w3),
            "unique_fraction_k4": _unique_fraction(w4),
            "redundancy_score_k2": _redundancy_score(w2),
            "redundancy_score_k3": _redundancy_score(w3),
            "redundancy_score_k4": _redundancy_score(w4),
            "top_count_k2": _top_count(w2),
            "top_count_k3": _top_count(w3),
            "top_freq_k2": _top_freq(w2),
            "top_freq_k3": _top_freq(w3),
            "duplicate_window_fraction_w5": _dup_window_frac(seq, 5),
            "duplicate_window_fraction_w6": _dup_window_frac(seq, 6),
            "repeated_word_burden_k2": _repeated_burden(seq, 2),
            "repeated_word_burden_k3": _repeated_burden(seq, 3),
            "recurrence_entropy_k2": _recurrence_entropy(w2),
            "recurrence_entropy_k3": _recurrence_entropy(w3),
            "repeated_span_k2": _repeated_span(seq, 2),
            "repeated_span_k3": _repeated_span(seq, 3),
            "repeated_span_norm_k2": _span_norm(seq, 2),
            "repeated_span_norm_k3": _span_norm(seq, 3),
            "repeated_block_density_k2": _repeated_block_density(seq, 2),
            "repeated_block_density_k3": _repeated_block_density(seq, 3),
            "longest_redundant_block_k2": _longest_redundant_block(seq, 2),
            "longest_redundant_block_k3": _longest_redundant_block(seq, 3),
            "recurrence_concentration_k2": _recurrence_concentration(w2),
            "recurrence_concentration_k3": _recurrence_concentration(w3),
            "local_redundancy_mean_w8_k2": _local_redundancy_mean(seq, 8, 2),
            "local_redundancy_std_w8_k2": _local_redundancy_std(seq, 8, 2),
            "local_redundancy_mean_w10_k2": _local_redundancy_mean(seq, 10, 2),
            "local_redundancy_std_w10_k2": _local_redundancy_std(seq, 10, 2),
        }
