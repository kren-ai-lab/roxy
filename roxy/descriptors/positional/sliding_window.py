"""Sliding-window profile descriptors."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from roxy.core.constants import AA_GROUPS, KD, POLARITY
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan

_HYDROPATHY_HIGH_THRESHOLD = 1.0
_CHARGED_HIGH_THRESHOLD = 0.4
_AROMATIC_HIGH_THRESHOLD = 0.2
_LOW_ENTROPY_THRESHOLD = 1.5

_CHARGED = frozenset(AA_GROUPS["charged"])
_AROMATIC = frozenset(AA_GROUPS["aromatic"])


def _windows(seq: str, size: int) -> list[str]:
    """Return all sliding windows of given size."""
    if len(seq) < size:
        return []
    return [seq[i : i + size] for i in range(len(seq) - size + 1)]


def _scale_mean(seq: str, scale: dict[str, float]) -> float:
    """Return mean scale value over sequence."""
    if not seq:
        return _NAN
    return float(np.mean([scale[aa] for aa in seq]))


def _group_fraction(seq: str, group: frozenset[str]) -> float:
    """Return fraction of residues in group."""
    if not seq:
        return _NAN
    return sum(aa in group for aa in seq) / len(seq)


def _shannon_entropy(seq: str) -> float:
    """Return Shannon entropy (bits) of residue distribution."""
    if not seq:
        return _NAN
    counts = Counter(seq)
    probs = np.array([c / len(seq) for c in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def _profile_stats(values: list[float]) -> dict[str, float]:
    """Return 6 summary stats for a window profile."""
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


def _fraction_above(values: list[float], threshold: float) -> float:
    """Return fraction of values strictly greater than threshold."""
    if not values:
        return _NAN
    return float(np.mean(np.array(values) > threshold))


@register("sliding_window", family="positional")
class SlidingWindowDescriptor(BaseDescriptor):
    """Sliding-window profile statistics for physicochemical properties.

    For each window size, five residue-level profiles are computed
    (hydropathy, polarity, charged fraction, aromatic fraction, Shannon
    entropy) and summarised with 6 statistics each, plus 4 threshold
    fraction features.  Columns use prefix ``win{size}_``.

    Args:
        window_sizes: Sizes of the sliding windows. Default ``(5, 7, 9)``.

    Output columns (prefix ``sliding_window_``):
        ``length``, ``valid_residue_count``,
        per window size: ``win{w}_{profile}_{stat}`` (5 profiles x 6 stats)
        and ``win{w}_hydropathy_high_fraction``,
        ``win{w}_charged_high_fraction``,
        ``win{w}_aromatic_high_fraction``,
        ``win{w}_low_entropy_fraction``.

    """

    def __init__(self, *, window_sizes: tuple[int, ...] = (5, 7, 9)) -> None:
        """Initialize SlidingWindowDescriptor."""
        self.window_sizes = tuple(window_sizes)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        _stat_keys = ("mean", "std", "min", "max", "amplitude", "start_end_diff")
        _profiles = ("hydropathy", "polarity", "charged_frac", "aromatic_frac", "entropy")
        _threshold_keys = (
            "hydropathy_high_fraction",
            "charged_high_fraction",
            "aromatic_high_fraction",
            "low_entropy_fraction",
        )
        for w in self.window_sizes:
            for profile in _profiles:
                for stat in _stat_keys:
                    feats[f"win{w}_{profile}_{stat}"] = _NAN
            for key in _threshold_keys:
                feats[f"win{w}_{key}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute sliding-window features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        for w in self.window_sizes:
            ws = _windows(seq, w)

            hydro = [_scale_mean(win, KD) for win in ws]
            polarity = [_scale_mean(win, POLARITY) for win in ws]
            charged = [_group_fraction(win, _CHARGED) for win in ws]
            aromatic = [_group_fraction(win, _AROMATIC) for win in ws]
            entropy = [_shannon_entropy(win) for win in ws]

            profiles = {
                "hydropathy": hydro,
                "polarity": polarity,
                "charged_frac": charged,
                "aromatic_frac": aromatic,
                "entropy": entropy,
            }

            for name, values in profiles.items():
                for stat, val in _profile_stats(values).items():
                    feats[f"win{w}_{name}_{stat}"] = val

            feats[f"win{w}_hydropathy_high_fraction"] = _fraction_above(hydro, _HYDROPATHY_HIGH_THRESHOLD)
            feats[f"win{w}_charged_high_fraction"] = _fraction_above(charged, _CHARGED_HIGH_THRESHOLD)
            feats[f"win{w}_aromatic_high_fraction"] = _fraction_above(aromatic, _AROMATIC_HIGH_THRESHOLD)
            feats[f"win{w}_low_entropy_fraction"] = _fraction_above(
                [-v for v in entropy], -_LOW_ENTROPY_THRESHOLD
            )

        return feats
