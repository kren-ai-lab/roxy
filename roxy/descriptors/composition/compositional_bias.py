"""Compositional bias and inequality descriptor."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from roxy.core.constants import AA20, AA_GROUPS
from roxy.descriptors._utils import clean_sequence, safe_ratio
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_AA20_LIST: list[str] = sorted(AA20)
_UNIFORM: float = 1.0 / 20

_GROUP_PAIRS: list[tuple[str, str]] = [
    ("positive", "negative"),
    ("hydrophobic", "hydrophilic"),
    ("polar", "nonpolar"),
    ("aromatic", "aliphatic"),
    ("disorder_promoting", "order_promoting"),
]


def _gini_like(freqs: np.ndarray) -> float:
    """Compute modified Gini-like inequality index on a frequency vector."""
    n = len(freqs)
    total = freqs.sum()
    if total == 0:
        return _NAN
    sorted_f = np.sort(freqs)
    idx = np.arange(1, n + 1, dtype=float)
    return float((2 * (idx * sorted_f).sum() / (n * total)) - (n + 1) / n)


def _kl_div_uniform(freqs: np.ndarray) -> float:
    """KL divergence from uniform distribution (bits)."""
    kl = 0.0
    for p in freqs:
        if p > 0:
            kl += p * math.log2(p / _UNIFORM)
    return kl


@register("compositional_bias", family="composition")
class CompositionalBiasDescriptor(BaseDescriptor):
    """Inequality and asymmetry measures of amino-acid usage.

    Computes global statistics on the 20-AA frequency vector plus
    group-level skew/ratio pairs for 5 physicochemical contrasts.

    Output columns (prefix ``compositional_bias_``):
        ``length``, ``valid_residue_count``, ``usage_mean/std/var/cv``,
        ``max/min_residue_fraction``, ``top2/3/5_burden``,
        ``dominance_gap``, ``gini_like_inequality``, ``kl_div_uniform``,
        ``l1_dev_uniform``, plus 5 ``*_skew`` and 5 ``*_ratio`` columns.

    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute compositional bias features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)
        empty = n == 0

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if empty:
            for key in (
                "usage_mean", "usage_std", "usage_var", "usage_cv",
                "max_residue_fraction", "min_residue_fraction",
                "top2_burden", "top3_burden", "top5_burden",
                "dominance_gap", "gini_like_inequality",
                "kl_div_uniform", "l1_dev_uniform",
            ):
                feats[key] = _NAN
            for a, b in _GROUP_PAIRS:
                feats[f"{a}_{b}_skew"] = _NAN
                feats[f"{a}_{b}_ratio"] = _NAN
            return feats

        counts = Counter(seq)
        freqs = np.array([counts.get(aa, 0) / n for aa in _AA20_LIST], dtype=float)
        sorted_desc = np.sort(freqs)[::-1]
        mean = float(np.mean(freqs))

        feats["usage_mean"] = mean
        feats["usage_std"] = float(np.std(freqs, ddof=0))
        feats["usage_var"] = float(np.var(freqs, ddof=0))
        feats["usage_cv"] = float(np.std(freqs, ddof=0) / mean) if mean != 0 else _NAN
        feats["max_residue_fraction"] = float(sorted_desc[0])
        feats["min_residue_fraction"] = float(sorted_desc[-1])
        feats["top2_burden"] = float(sorted_desc[:2].sum())
        feats["top3_burden"] = float(sorted_desc[:3].sum())
        feats["top5_burden"] = float(sorted_desc[:5].sum())
        feats["dominance_gap"] = float(sorted_desc[0] - sorted_desc[1]) if len(sorted_desc) >= 2 else _NAN  # noqa: PLR2004
        feats["gini_like_inequality"] = _gini_like(freqs)
        feats["kl_div_uniform"] = _kl_div_uniform(freqs)
        feats["l1_dev_uniform"] = float(np.abs(freqs - _UNIFORM).sum())

        for a, b in _GROUP_PAIRS:
            fa = sum(aa in AA_GROUPS[a] for aa in seq) / n
            fb = sum(aa in AA_GROUPS[b] for aa in seq) / n
            feats[f"{a}_{b}_skew"] = fa - fb
            feats[f"{a}_{b}_ratio"] = safe_ratio(fa, fb)

        return feats
