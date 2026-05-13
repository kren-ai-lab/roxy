"""Hydrophobicity, polarity, and amphipathicity descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS, KD, POLARITY
from roxy.descriptors._utils import (
    fraction_above_threshold,
    fraction_from_group,
    membership_array,
    profile_stats,
    rolling_mean,
    scale_array,
    scale_mean,
    scale_std,
    terminal_segment,
)
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_HYDROPHOBIC = AA_GROUPS["hydrophobic"]
_HYDROPHILIC = AA_GROUPS["hydrophilic"]
_POLAR = AA_GROUPS["polar"]
_NONPOLAR = AA_GROUPS["nonpolar"]
_AROMATIC = AA_GROUPS["aromatic"]

_HYDROPHOBIC_THRESHOLD = 0.6
_POLAR_THRESHOLD = 0.6
_AMPHIPATHICITY_THRESHOLD = 1.0


def _local_contrast_profile(seq: str, window: int) -> list[float]:
    """Return per-window |hydrophobic_frac - polar_frac| contrast."""
    hydrophobic = rolling_mean(membership_array(seq, _HYDROPHOBIC), window)
    polar = rolling_mean(membership_array(seq, _POLAR), window)
    return list(np.abs(hydrophobic - polar))


def _local_amphipathicity_profile(seq: str, window: int) -> list[float]:
    """Return per-window amphipathicity proxy: |hf-pf| * |hydro_mean - polarity_mean|."""
    hydrophobic = rolling_mean(membership_array(seq, _HYDROPHOBIC), window)
    polar = rolling_mean(membership_array(seq, _POLAR), window)
    hydropathy = rolling_mean(scale_array(seq, KD), window)
    polarity = rolling_mean(scale_array(seq, POLARITY), window)
    return list(np.abs(hydrophobic - polar) * np.abs(hydropathy - polarity))


@register("hydrophobicity", family="physicochemical")
class HydrophobicityDescriptor(BaseDescriptor):
    """Hydrophobicity, polarity, and amphipathicity features.

    Computes global scale statistics, group fractions, terminal asymmetry,
    and windowed local profiles at configurable window sizes.

    Args:
        window_sizes: Tuple of window sizes for local profiles.
        terminal_window: Residue count for N/C-terminal segments.

    Output columns (prefix ``hydrophobicity_``):
        ``length``, ``valid_residue_count``,
        ``hydropathy_mean/std``, ``polarity_mean/std``,
        ``hydrophobic/hydrophilic/polar/nonpolar/aromatic_fraction``,
        ``hydrophobic_hydrophilic_balance``, ``polar_nonpolar_balance``,
        ``global_amphipathicity_proxy``,
        ``nterm/cterm_hydropathy/polarity_mean``,
        ``terminal_hydropathy/polarity_asymmetry``,
        per-window: ``w{N}_{profile}_{stat}`` and ``w{N}_{patch}_fraction``.

    """

    def __init__(
        self,
        *,
        window_sizes: tuple[int, ...] = (5, 7, 9),
        terminal_window: int = 10,
    ) -> None:
        """Initialize HydrophobicityDescriptor."""
        self.window_sizes = tuple(window_sizes)
        self.terminal_window = terminal_window

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for key in (
            "hydropathy_mean",
            "hydropathy_std",
            "polarity_mean",
            "polarity_std",
            "hydrophobic_fraction",
            "hydrophilic_fraction",
            "polar_fraction",
            "nonpolar_fraction",
            "aromatic_fraction",
            "hydrophobic_hydrophilic_balance",
            "polar_nonpolar_balance",
            "global_amphipathicity_proxy",
            "nterm_hydropathy_mean",
            "cterm_hydropathy_mean",
            "nterm_polarity_mean",
            "cterm_polarity_mean",
            "terminal_hydropathy_asymmetry",
            "terminal_polarity_asymmetry",
        ):
            feats[key] = _NAN
        for ws in self.window_sizes:
            for profile in (
                "hydropathy",
                "polarity",
                "hydrophobic_frac",
                "polar_frac",
                "contrast",
                "amphipathicity",
            ):
                for stat in ("mean", "std", "min", "max", "amplitude", "start_end_diff"):
                    feats[f"w{ws}_{profile}_{stat}"] = _NAN
            feats[f"w{ws}_hydrophobic_patch_fraction"] = _NAN
            feats[f"w{ws}_polar_patch_fraction"] = _NAN
            feats[f"w{ws}_high_amphipathicity_fraction"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute hydrophobicity/polarity/amphipathicity features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        feats["hydropathy_mean"] = scale_mean(seq, KD)
        feats["hydropathy_std"] = scale_std(seq, KD)
        feats["polarity_mean"] = scale_mean(seq, POLARITY)
        feats["polarity_std"] = scale_std(seq, POLARITY)

        feats["hydrophobic_fraction"] = fraction_from_group(seq, _HYDROPHOBIC)
        feats["hydrophilic_fraction"] = fraction_from_group(seq, _HYDROPHILIC)
        feats["polar_fraction"] = fraction_from_group(seq, _POLAR)
        feats["nonpolar_fraction"] = fraction_from_group(seq, _NONPOLAR)
        feats["aromatic_fraction"] = fraction_from_group(seq, _AROMATIC)

        feats["hydrophobic_hydrophilic_balance"] = (
            feats["hydrophobic_fraction"] - feats["hydrophilic_fraction"]
        )
        feats["polar_nonpolar_balance"] = feats["polar_fraction"] - feats["nonpolar_fraction"]
        feats["global_amphipathicity_proxy"] = abs(
            feats["hydrophobic_fraction"] - feats["polar_fraction"]
        ) * abs(feats["hydropathy_mean"] - feats["polarity_mean"])

        nterm = terminal_segment(seq, "N", self.terminal_window)
        cterm = terminal_segment(seq, "C", self.terminal_window)
        feats["nterm_hydropathy_mean"] = scale_mean(nterm, KD)
        feats["cterm_hydropathy_mean"] = scale_mean(cterm, KD)
        feats["nterm_polarity_mean"] = scale_mean(nterm, POLARITY)
        feats["cterm_polarity_mean"] = scale_mean(cterm, POLARITY)
        feats["terminal_hydropathy_asymmetry"] = (
            feats["nterm_hydropathy_mean"] - feats["cterm_hydropathy_mean"]
        )
        feats["terminal_polarity_asymmetry"] = feats["nterm_polarity_mean"] - feats["cterm_polarity_mean"]

        hydropathy_values = scale_array(seq, KD)
        polarity_values = scale_array(seq, POLARITY)
        hydrophobic_mask = membership_array(seq, _HYDROPHOBIC)
        polar_mask = membership_array(seq, _POLAR)

        for ws in self.window_sizes:
            hydro_profile = rolling_mean(hydropathy_values, ws)
            polarity_profile = rolling_mean(polarity_values, ws)
            hydrophobic_frac_profile = rolling_mean(hydrophobic_mask, ws)
            polar_frac_profile = rolling_mean(polar_mask, ws)
            contrast_profile = np.abs(hydrophobic_frac_profile - polar_frac_profile)
            amphi_profile = contrast_profile * np.abs(hydro_profile - polarity_profile)

            for profile_name, values in (
                ("hydropathy", hydro_profile),
                ("polarity", polarity_profile),
                ("hydrophobic_frac", hydrophobic_frac_profile),
                ("polar_frac", polar_frac_profile),
                ("contrast", contrast_profile),
                ("amphipathicity", amphi_profile),
            ):
                stats = profile_stats(values)
                prefix = f"w{ws}_{profile_name}"
                for stat, val in stats.items():
                    feats[f"{prefix}_{stat}"] = val

            feats[f"w{ws}_hydrophobic_patch_fraction"] = fraction_above_threshold(
                hydrophobic_frac_profile, _HYDROPHOBIC_THRESHOLD
            )
            feats[f"w{ws}_polar_patch_fraction"] = fraction_above_threshold(
                polar_frac_profile, _POLAR_THRESHOLD
            )
            feats[f"w{ws}_high_amphipathicity_fraction"] = fraction_above_threshold(
                amphi_profile, _AMPHIPATHICITY_THRESHOLD
            )

        return feats
