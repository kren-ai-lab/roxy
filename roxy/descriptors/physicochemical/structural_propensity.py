"""Structural propensity (Chou-Fasman) descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import CF_HELIX, CF_SHEET, CF_TURN
from roxy.descriptors._utils import (
    fraction_above_threshold,
    fraction_from_group,
    profile_stats,
    rolling_mean,
    scale_array,
)
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_HELIX_FAVORING = frozenset(aa for aa, v in CF_HELIX.items() if v >= 1.0)
_SHEET_FAVORING = frozenset(aa for aa, v in CF_SHEET.items() if v >= 1.0)
_TURN_FAVORING = frozenset(aa for aa, v in CF_TURN.items() if v >= 1.0)


@register("structural_propensity", family="physicochemical")
class StructuralPropensityDescriptor(BaseDescriptor):
    """Chou-Fasman structural propensity features.

    Computes global helix/sheet/turn propensity statistics, favoring-residue
    fractions, and windowed local propensity profiles.

    Args:
        window_sizes: Tuple of window sizes for local propensity profiles.
        threshold: Propensity value above which a window is "high-propensity".

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``structural_propensity_``.
        ``length``, ``valid_residue_count``,
        ``helix/sheet/turn_mean/std``,
        ``helix/sheet/turn_favoring_fraction``,
        ``helix_sheet_balance``, ``turn_vs_secondary_balance``,
        per-window: ``w{N}_{helix/sheet/turn}_{stat}``,
        ``w{N}_{helix/sheet/turn}_high_fraction``,
        ``w{N}_helix_sheet_balance_{mean/max/min}``.

    """

    def __init__(
        self,
        *,
        window_sizes: tuple[int, ...] = (5, 7),
        threshold: float = 1.0,
    ) -> None:
        """Initialize StructuralPropensityDescriptor."""
        self.window_sizes = tuple(window_sizes)
        self.threshold = threshold

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for key in (
            "helix_mean",
            "helix_std",
            "sheet_mean",
            "sheet_std",
            "turn_mean",
            "turn_std",
            "helix_favoring_fraction",
            "sheet_favoring_fraction",
            "turn_favoring_fraction",
            "helix_sheet_balance",
            "turn_vs_secondary_balance",
        ):
            feats[key] = _NAN
        for ws in self.window_sizes:
            for ss in ("helix", "sheet", "turn"):
                for stat in ("mean", "std", "min", "max", "amplitude"):
                    feats[f"w{ws}_{ss}_{stat}"] = _NAN
                feats[f"w{ws}_{ss}_high_fraction"] = _NAN
            for stat in ("mean", "max", "min"):
                feats[f"w{ws}_helix_sheet_balance_{stat}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute structural propensity features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        helix_vals = [CF_HELIX[aa] for aa in seq]
        sheet_vals = [CF_SHEET[aa] for aa in seq]
        turn_vals = [CF_TURN[aa] for aa in seq]

        feats["helix_mean"] = float(np.mean(helix_vals))
        feats["sheet_mean"] = float(np.mean(sheet_vals))
        feats["turn_mean"] = float(np.mean(turn_vals))
        feats["helix_std"] = float(np.std(helix_vals, ddof=0))
        feats["sheet_std"] = float(np.std(sheet_vals, ddof=0))
        feats["turn_std"] = float(np.std(turn_vals, ddof=0))

        feats["helix_favoring_fraction"] = fraction_from_group(seq, _HELIX_FAVORING)
        feats["sheet_favoring_fraction"] = fraction_from_group(seq, _SHEET_FAVORING)
        feats["turn_favoring_fraction"] = fraction_from_group(seq, _TURN_FAVORING)

        feats["helix_sheet_balance"] = feats["helix_mean"] - feats["sheet_mean"]
        feats["turn_vs_secondary_balance"] = (
            feats["turn_mean"] - (feats["helix_mean"] + feats["sheet_mean"]) / 2.0
        )

        helix_arr = scale_array(seq, CF_HELIX)
        sheet_arr = scale_array(seq, CF_SHEET)
        turn_arr = scale_array(seq, CF_TURN)

        for ws in self.window_sizes:
            helix_profile = rolling_mean(helix_arr, ws)
            sheet_profile = rolling_mean(sheet_arr, ws)
            turn_profile = rolling_mean(turn_arr, ws)

            for ss_name, values in (
                ("helix", helix_profile),
                ("sheet", sheet_profile),
                ("turn", turn_profile),
            ):
                stats = profile_stats(values)
                for stat in ("mean", "std", "min", "max", "amplitude"):
                    feats[f"w{ws}_{ss_name}_{stat}"] = stats[stat]
                feats[f"w{ws}_{ss_name}_high_fraction"] = fraction_above_threshold(values, self.threshold)

            if helix_profile.size:
                balance = helix_profile - sheet_profile
                feats[f"w{ws}_helix_sheet_balance_mean"] = float(np.mean(balance))
                feats[f"w{ws}_helix_sheet_balance_max"] = float(np.max(balance))
                feats[f"w{ws}_helix_sheet_balance_min"] = float(np.min(balance))
            else:
                feats[f"w{ws}_helix_sheet_balance_mean"] = _NAN
                feats[f"w{ws}_helix_sheet_balance_max"] = _NAN
                feats[f"w{ws}_helix_sheet_balance_min"] = _NAN

        return feats
