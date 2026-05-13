"""Residue spacing and inter-event distance descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_TRACKED_GROUPS: dict[str, frozenset[str]] = {
    "charged": frozenset(AA_GROUPS["charged"]),
    "hydrophobic": frozenset(AA_GROUPS["hydrophobic"]),
    "aromatic": frozenset(AA_GROUPS["aromatic"]),
    "polar": frozenset(AA_GROUPS["polar"]),
    "positive": frozenset(AA_GROUPS["positive"]),
    "negative": frozenset(AA_GROUPS["negative"]),
    "gly": frozenset("G"),
    "pro": frozenset("P"),
}

_CROSS_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("positive_negative_cross_mean", "positive", "negative"),
    ("charged_hydrophobic_cross_mean", "charged", "hydrophobic"),
    ("aromatic_polar_cross_mean", "aromatic", "polar"),
    ("disorder_order_cross_mean", "disorder_promoting", "order_promoting"),
)

_DISORDER = frozenset(AA_GROUPS["disorder_promoting"])
_ORDER = frozenset(AA_GROUPS["order_promoting"])

_MIN_PAIR = 2


def _positions(seq: str, group: frozenset[str]) -> list[int]:
    return [i for i, aa in enumerate(seq) if aa in group]


def _inter_event(positions: list[int]) -> list[int]:
    if len(positions) < _MIN_PAIR:
        return []
    return list(np.diff(positions))


def _nearest_neighbor(positions: list[int]) -> list[int]:
    if len(positions) < _MIN_PAIR:
        return []
    diffs = np.diff(np.array(positions, dtype=int))
    if diffs.size == 1:
        return [int(diffs[0]), int(diffs[0])]
    interior = np.minimum(diffs[:-1], diffs[1:])
    return [int(diffs[0]), *[int(v) for v in interior], int(diffs[-1])]


def _safe_stats(values: list) -> dict[str, float]:
    if not values:
        return {"mean": _NAN, "median": _NAN, "min": _NAN, "max": _NAN, "std": _NAN}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def _norm(value: float, seq_len: int) -> float:
    if seq_len == 0 or math.isnan(value):
        return _NAN
    return value / seq_len


def _cross_mean(seq: str, group_a: frozenset[str], group_b: frozenset[str]) -> float:
    pos_a = _positions(seq, group_a)
    pos_b = _positions(seq, group_b)
    if not pos_a or not pos_b:
        return _NAN
    a = np.array(pos_a, dtype=int)
    b = np.array(pos_b, dtype=int)
    idx = np.searchsorted(b, a)
    right = np.where(idx < b.size, np.abs(b[np.clip(idx, 0, b.size - 1)] - a), np.inf)
    left_idx = idx - 1
    left = np.where(left_idx >= 0, np.abs(b[np.clip(left_idx, 0, b.size - 1)] - a), np.inf)
    return float(np.minimum(left, right).mean())


_GROUP_STAT_KEYS = (
    "count",
    "density",
    "span",
    "span_norm",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "mean_norm",
    "median_norm",
    "max_norm",
    "nn_mean",
    "nn_median",
    "nn_min",
    "nn_max",
)
_CROSS_KEYS = tuple(t[0] for t in _CROSS_PAIRS)


@register("spacing", family="motif")
class SpacingDescriptor(BaseDescriptor):
    """Inter-residue spacing and nearest-neighbour distance descriptors.

    For each tracked group, computes count, density, span, inter-event
    distances (mean/median/min/max/std and normalised variants), and
    nearest-neighbour distances.  Also computes 4 cross-group mean
    distances.

    Output columns (prefix ``spacing_``):
        ``length``, ``valid_residue_count``,
        per group (8): 16 statistics each,
        4 cross-group ``{a}_{b}_cross_mean`` columns.
        Total: 2 + 8*16 + 4 = 134 columns.
    """

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for name in _TRACKED_GROUPS:
            for key in _GROUP_STAT_KEYS:
                val = 0.0 if key == "count" else _NAN
                feats[f"{name}_{key}"] = val
        for key in _CROSS_KEYS:
            feats[key] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute spacing features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        for name, group in _TRACKED_GROUPS.items():
            pos = _positions(seq, group)
            dists = _inter_event(pos)
            nn = _nearest_neighbor(pos)
            dist_s = _safe_stats(dists)
            nn_s = _safe_stats(nn)
            span = float(pos[-1] - pos[0]) if len(pos) >= _MIN_PAIR else _NAN

            feats[f"{name}_count"] = float(len(pos))
            feats[f"{name}_density"] = float(len(pos)) / n
            feats[f"{name}_span"] = span
            feats[f"{name}_span_norm"] = _norm(span, n)
            feats[f"{name}_mean"] = dist_s["mean"]
            feats[f"{name}_median"] = dist_s["median"]
            feats[f"{name}_min"] = dist_s["min"]
            feats[f"{name}_max"] = dist_s["max"]
            feats[f"{name}_std"] = dist_s["std"]
            feats[f"{name}_mean_norm"] = _norm(dist_s["mean"], n)
            feats[f"{name}_median_norm"] = _norm(dist_s["median"], n)
            feats[f"{name}_max_norm"] = _norm(dist_s["max"], n)
            feats[f"{name}_nn_mean"] = nn_s["mean"]
            feats[f"{name}_nn_median"] = nn_s["median"]
            feats[f"{name}_nn_min"] = nn_s["min"]
            feats[f"{name}_nn_max"] = nn_s["max"]

        _group_sets: dict[str, frozenset[str]] = {
            "positive": frozenset(AA_GROUPS["positive"]),
            "negative": frozenset(AA_GROUPS["negative"]),
            "charged": frozenset(AA_GROUPS["charged"]),
            "hydrophobic": frozenset(AA_GROUPS["hydrophobic"]),
            "aromatic": frozenset(AA_GROUPS["aromatic"]),
            "polar": frozenset(AA_GROUPS["polar"]),
            "disorder_promoting": _DISORDER,
            "order_promoting": _ORDER,
        }
        for key, a_name, b_name in _CROSS_PAIRS:
            feats[key] = _cross_mean(seq, _group_sets[a_name], _group_sets[b_name])

        return feats
