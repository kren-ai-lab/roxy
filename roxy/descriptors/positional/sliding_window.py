"""Sliding-window profile descriptors."""

from __future__ import annotations

import math

from roxy.core.constants import AA_GROUPS, KD, POLARITY
from roxy.descriptors._utils import (
    clean_sequence,
    fraction_above_threshold,
    fraction_from_group,
    profile_stats,
    scale_mean,
    shannon_entropy,
    windows,
)
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_HYDROPATHY_HIGH_THRESHOLD = 1.0
_CHARGED_HIGH_THRESHOLD = 0.4
_AROMATIC_HIGH_THRESHOLD = 0.2
_LOW_ENTROPY_THRESHOLD = 1.5

_CHARGED = frozenset(AA_GROUPS["charged"])
_AROMATIC = frozenset(AA_GROUPS["aromatic"])


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
            ws = windows(seq, w)

            hydro = [scale_mean(win, KD) for win in ws]
            polarity = [scale_mean(win, POLARITY) for win in ws]
            charged = [fraction_from_group(win, _CHARGED) for win in ws]
            aromatic = [fraction_from_group(win, _AROMATIC) for win in ws]
            entropy = [shannon_entropy(win) for win in ws]

            profiles = {
                "hydropathy": hydro,
                "polarity": polarity,
                "charged_frac": charged,
                "aromatic_frac": aromatic,
                "entropy": entropy,
            }

            for name, values in profiles.items():
                for stat, val in profile_stats(values).items():
                    feats[f"win{w}_{name}_{stat}"] = val

            feats[f"win{w}_hydropathy_high_fraction"] = fraction_above_threshold(
                hydro, _HYDROPATHY_HIGH_THRESHOLD
            )
            feats[f"win{w}_charged_high_fraction"] = fraction_above_threshold(
                charged, _CHARGED_HIGH_THRESHOLD
            )
            feats[f"win{w}_aromatic_high_fraction"] = fraction_above_threshold(
                aromatic, _AROMATIC_HIGH_THRESHOLD
            )
            feats[f"win{w}_low_entropy_fraction"] = fraction_above_threshold(
                [-v for v in entropy], -_LOW_ENTROPY_THRESHOLD
            )

        return feats
