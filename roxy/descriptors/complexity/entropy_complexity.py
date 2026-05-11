"""Sequence complexity and low-complexity descriptors."""

from __future__ import annotations

import math
from collections import Counter
from itertools import groupby

import numpy as np

from roxy.core.constants import AA20
from roxy.descriptors._utils import clean_sequence, linguistic_complexity, shannon_entropy
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_AA20_LIST: list[str] = sorted(AA20)
_N_AA = len(_AA20_LIST)


def _homopolymer_burden(seq: str, min_run: int) -> float:
    runs = [len(list(g)) for _, g in groupby(seq)]
    burden = sum(r for r in runs if r >= min_run)
    return burden / len(seq)


def _gini_like(seq: str) -> float:
    n = len(seq)
    counts = Counter(seq)
    freqs = np.array(sorted(counts.get(aa, 0) / n for aa in _AA20_LIST), dtype=float)
    total = freqs.sum()
    if total == 0:
        return _NAN
    idx = np.arange(1, _N_AA + 1, dtype=float)
    return float((2 * np.sum(idx * freqs) / (_N_AA * total)) - (_N_AA + 1) / _N_AA)


@register("entropy_complexity", family="complexity")
class EntropyComplexityDescriptor(BaseDescriptor):
    """Sequence complexity and low-complexity region features.

    Covers entropy measures, linguistic complexity, k-mer repetition,
    homopolymer runs, windowed entropy profiles, and compositional inequality.

    Args:
        low_complexity_window: Window size for entropy-based low-complexity scan.
        entropy_threshold: Shannon entropy threshold below which a window is
            considered low-complexity.

    Output columns (prefix ``entropy_complexity_``):
        ``length``, ``valid_residue_count``,
        ``shannon_entropy``, ``shannon_entropy_norm``,
        ``linguistic_complexity_k{1/2/3}``,
        ``unique_kmer_fraction_k{2/3}``, ``repeated_kmer_fraction_k{2/3}``,
        ``longest_homopolymer_run``, ``homopolymer_burden_len{2/3}``,
        ``low_complexity_window_fraction_w{N}``,
        ``window_entropy_mean_w{N}``, ``window_entropy_std_w{N}``,
        ``most_frequent_residue_fraction``,
        ``gini_like_inequality``, ``residue_dominance_gap``.

    """

    def __init__(
        self,
        *,
        low_complexity_window: int = 5,
        entropy_threshold: float = 1.5,
    ) -> None:
        """Initialize EntropyComplexityDescriptor."""
        self.low_complexity_window = low_complexity_window
        self.entropy_threshold = entropy_threshold

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        w = self.low_complexity_window
        for key in (
            "shannon_entropy",
            "shannon_entropy_norm",
            "linguistic_complexity_k1",
            "linguistic_complexity_k2",
            "linguistic_complexity_k3",
            "unique_kmer_fraction_k2",
            "unique_kmer_fraction_k3",
            "repeated_kmer_fraction_k2",
            "repeated_kmer_fraction_k3",
            "homopolymer_burden_len2",
            "homopolymer_burden_len3",
            f"low_complexity_window_fraction_w{w}",
            f"window_entropy_mean_w{w}",
            f"window_entropy_std_w{w}",
            "most_frequent_residue_fraction",
            "gini_like_inequality",
            "residue_dominance_gap",
        ):
            feats[key] = _NAN
        feats["longest_homopolymer_run"] = 0.0
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute complexity features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)
        w = self.low_complexity_window

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        feats["shannon_entropy"] = shannon_entropy(seq)
        max_h = math.log2(min(n, _N_AA))
        feats["shannon_entropy_norm"] = feats["shannon_entropy"] / max_h if max_h > 0 else _NAN

        for k in (1, 2, 3):
            feats[f"linguistic_complexity_k{k}"] = linguistic_complexity(seq, k)

        for k in (2, 3):
            words = [seq[i : i + k] for i in range(n - k + 1)] if n >= k else []
            if not words:
                feats[f"unique_kmer_fraction_k{k}"] = _NAN
                feats[f"repeated_kmer_fraction_k{k}"] = _NAN
            else:
                counts = Counter(words)
                feats[f"unique_kmer_fraction_k{k}"] = len(set(words)) / len(words)
                feats[f"repeated_kmer_fraction_k{k}"] = sum(v for v in counts.values() if v > 1) / len(words)

        runs = [len(list(g)) for _, g in groupby(seq)]
        feats["longest_homopolymer_run"] = float(max(runs))
        feats["homopolymer_burden_len2"] = _homopolymer_burden(seq, 2)
        feats["homopolymer_burden_len3"] = _homopolymer_burden(seq, 3)

        win_list = [seq[i : i + w] for i in range(n - w + 1)] if n >= w else []
        if not win_list:
            feats[f"low_complexity_window_fraction_w{w}"] = _NAN
            feats[f"window_entropy_mean_w{w}"] = _NAN
            feats[f"window_entropy_std_w{w}"] = _NAN
        else:
            entropies = [shannon_entropy(ww) for ww in win_list]
            feats[f"low_complexity_window_fraction_w{w}"] = sum(
                e <= self.entropy_threshold for e in entropies
            ) / len(entropies)
            feats[f"window_entropy_mean_w{w}"] = float(np.mean(entropies))
            feats[f"window_entropy_std_w{w}"] = float(np.std(entropies, ddof=0))

        counts = Counter(seq)
        most_common = counts.most_common()
        feats["most_frequent_residue_fraction"] = most_common[0][1] / n
        feats["gini_like_inequality"] = _gini_like(seq)
        feats["residue_dominance_gap"] = (
            (most_common[0][1] - most_common[1][1]) / n if len(most_common) > 1 else 1.0
        )

        return feats
