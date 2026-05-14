"""Sliding-window profile descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS, KD, POLARITY
from roxy.descriptors._utils import (
    fraction_above_threshold,
    membership_array,
    profile_stats,
    rolling_mean,
    scale_array,
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

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``sliding_window_``.
        ``length``, ``valid_residue_count``,
        per window size: ``win{w}_{profile}_{stat}``
        (5 profiles x 6 stats) and
        ``win{w}_hydropathy_high_fraction``,
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
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        hydropathy_values = scale_array(seq, KD)
        polarity_values = scale_array(seq, POLARITY)
        charged_mask = membership_array(seq, _CHARGED)
        aromatic_mask = membership_array(seq, _AROMATIC)

        for w in self.window_sizes:
            ws = windows(seq, w)

            hydro = rolling_mean(hydropathy_values, w)
            polarity = rolling_mean(polarity_values, w)
            charged = rolling_mean(charged_mask, w)
            aromatic = rolling_mean(aromatic_mask, w)
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
                -np.asarray(entropy, dtype=float), -_LOW_ENTROPY_THRESHOLD
            )

        return feats
