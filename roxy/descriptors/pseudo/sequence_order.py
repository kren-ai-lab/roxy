"""Sequence-order adjacency and transition descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS
from roxy.descriptors._utils import clean_sequence
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_MIN_LEN = 2
_CLUSTER_WINDOW = 3
_CLUSTER_MIN_HITS = 2

_TRACKED_GROUPS = (
    "charged",
    "hydrophobic",
    "polar",
    "aromatic",
    "disorder_promoting",
    "order_promoting",
)

_CROSS_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("charged_hydrophobic_transition_frac", "charged", "hydrophobic"),
    ("polar_nonpolar_transition_frac", "polar", "nonpolar"),
    ("disorder_order_transition_frac", "disorder_promoting", "order_promoting"),
    ("positive_negative_transition_frac", "positive", "negative"),
)

_GROUP_STAT_SUFFIXES = (
    "same_adj_frac",
    "mean_spacing_norm",
    "cluster_frac_w3",
    "lag1_coupling",
    "lag2_coupling",
    "adj_enrichment",
)


def _adjacent_pairs(seq: str) -> list[tuple[str, str]]:
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _same_group_adj_frac(seq: str, group: frozenset[str]) -> float:
    if len(seq) < _MIN_LEN:
        return _NAN
    pairs = _adjacent_pairs(seq)
    same = sum((a in group) and (b in group) for a, b in pairs)
    return same / len(pairs)


def _cross_transition_frac(seq: str, g_a: frozenset[str], g_b: frozenset[str]) -> float:
    if len(seq) < _MIN_LEN:
        return _NAN
    pairs = _adjacent_pairs(seq)
    hits = sum((a in g_a and b in g_b) or (a in g_b and b in g_a) for a, b in pairs)
    return hits / len(pairs)


def _normalized_mean_spacing(seq: str, group: frozenset[str]) -> float:
    n = len(seq)
    if n == 0:
        return _NAN
    pos = [i for i, aa in enumerate(seq) if aa in group]
    if len(pos) < _MIN_LEN:
        return _NAN
    spacing = float(np.mean(np.diff(pos)))
    return spacing / n


def _cluster_frac(seq: str, group: frozenset[str]) -> float:
    n = len(seq)
    if n < _CLUSTER_WINDOW:
        return _NAN
    total = n - _CLUSTER_WINDOW + 1
    hits = sum(
        sum(aa in group for aa in seq[i : i + _CLUSTER_WINDOW]) >= _CLUSTER_MIN_HITS for i in range(total)
    )
    return hits / total


def _lag_coupling(seq: str, group: frozenset[str], lag: int) -> float:
    if len(seq) <= lag:
        return _NAN
    x = np.array([1 if aa in group else 0 for aa in seq], dtype=float)
    return float(np.mean(x[:-lag] * x[lag:]))


def _adj_enrichment(seq: str, group: frozenset[str]) -> float:
    if len(seq) < _MIN_LEN:
        return _NAN
    p = float(np.mean([aa in group for aa in seq]))
    observed = _same_group_adj_frac(seq, group)
    expected = p * p
    if expected == 0:
        return _NAN
    return observed / expected


@register("sequence_order", family="pseudo")
class SequenceOrderDescriptor(BaseDescriptor):
    """Sequence-order adjacency, clustering, and transition descriptors.

    Tracks 6 residue groups and computes same-group adjacency fraction,
    mean spacing (normalised), local cluster fraction (window=3),
    lag-1/lag-2 coupling, and adjacency enrichment.  Four cross-group
    transition fractions are also computed.

    Output columns (prefix ``sequence_order_``):
        ``length``, ``valid_residue_count``,
        per group (6): 6 statistics each,
        4 cross-group ``{a}_{b}_transition_frac`` columns.
        Total: 2 + 6*6 + 4 = 42 columns.
    """

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for name in _TRACKED_GROUPS:
            for suffix in _GROUP_STAT_SUFFIXES:
                feats[f"{name}_{suffix}"] = _NAN
        for key, _, _ in _CROSS_PAIRS:
            feats[key] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute sequence-order features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        for name in _TRACKED_GROUPS:
            group = frozenset(AA_GROUPS[name])
            feats[f"{name}_same_adj_frac"] = _same_group_adj_frac(seq, group)
            feats[f"{name}_mean_spacing_norm"] = _normalized_mean_spacing(seq, group)
            feats[f"{name}_cluster_frac_w3"] = _cluster_frac(seq, group)
            feats[f"{name}_lag1_coupling"] = _lag_coupling(seq, group, lag=1)
            feats[f"{name}_lag2_coupling"] = _lag_coupling(seq, group, lag=2)
            feats[f"{name}_adj_enrichment"] = _adj_enrichment(seq, group)

        for key, a_name, b_name in _CROSS_PAIRS:
            feats[key] = _cross_transition_frac(
                seq,
                frozenset(AA_GROUPS[a_name]),
                frozenset(AA_GROUPS[b_name]),
            )

        return feats
