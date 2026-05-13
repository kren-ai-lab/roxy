"""Residue positional distribution descriptors."""

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


def _bin_occupancy(norm_positions: list[float], n_bins: int) -> np.ndarray:
    """Fraction of positions per equal-width bin over [0, 1]."""
    if not norm_positions:
        return np.zeros(n_bins, dtype=float)
    counts, _ = np.histogram(norm_positions, bins=n_bins, range=(0.0, 1.0))
    return counts.astype(float) / len(norm_positions)


def _summarize(norm_positions: list[float], prefix: str, n_bins: int) -> dict[str, float]:
    occ = _bin_occupancy(norm_positions, n_bins)
    cum = np.cumsum(occ)
    uniform = 1.0 / n_bins
    out: dict[str, float] = {}

    for i, v in enumerate(occ, 1):
        out[f"{prefix}_bin{i}_frac"] = float(v)
    for i, v in enumerate(cum, 1):
        out[f"{prefix}_cum{i}"] = float(v)

    if occ.sum() == 0:
        out[f"{prefix}_entropy"] = _NAN
        out[f"{prefix}_max_bin_frac"] = _NAN
        out[f"{prefix}_spread"] = _NAN
        for i in range(n_bins):
            out[f"{prefix}_bin{i + 1}_enrichment"] = _NAN
    else:
        probs = occ[occ > 0]
        out[f"{prefix}_entropy"] = float(-(probs * np.log2(probs)).sum())
        out[f"{prefix}_max_bin_frac"] = float(occ.max())
        out[f"{prefix}_spread"] = float(np.sum(occ > 0) / n_bins)
        for i in range(n_bins):
            out[f"{prefix}_bin{i + 1}_enrichment"] = float(occ[i] - uniform)

    return out


@register("distribution", family="ctd")
class DistributionDescriptor(BaseDescriptor):
    """Positional distribution of residue groups across the sequence.

    For each tracked group, bins normalized residue positions into terciles
    and quartiles and computes bin fractions, cumulative distributions,
    entropy, concentration, spread, and per-bin enrichment over uniform.

    Args:
        groups: Mapping of group name → residue set. Defaults to 8 built-in
            groups (charged, hydrophobic, aromatic, polar, positive, negative,
            gly, pro).

    Output columns (prefix ``distribution_``):
        ``length``, ``valid_residue_count``,
        per group: ``{name}_count``,
        ``{name}_tercile_bin{1/2/3}_{frac/enrichment}``,
        ``{name}_tercile_cum{1/2/3}``,
        ``{name}_tercile_{entropy/max_bin_frac/spread}``,
        ``{name}_quartile_bin{1..4}_{frac/enrichment}``,
        ``{name}_quartile_cum{1..4}``,
        ``{name}_quartile_{entropy/max_bin_frac/spread}``.

    """

    def __init__(
        self,
        *,
        groups: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Initialize DistributionDescriptor."""
        self.groups = groups if groups is not None else dict(_TRACKED_GROUPS)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for name in self.groups:
            feats[f"{name}_count"] = _NAN
            for bins, label in ((3, "tercile"), (4, "quartile")):
                for i in range(1, bins + 1):
                    feats[f"{name}_{label}_bin{i}_frac"] = _NAN
                    feats[f"{name}_{label}_cum{i}"] = _NAN
                    feats[f"{name}_{label}_bin{i}_enrichment"] = _NAN
                feats[f"{name}_{label}_entropy"] = _NAN
                feats[f"{name}_{label}_max_bin_frac"] = _NAN
                feats[f"{name}_{label}_spread"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute distribution features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        for name, group in self.groups.items():
            positions = [i + 1 for i, aa in enumerate(seq) if aa in group]
            norm_pos = [p / n for p in positions]
            feats[f"{name}_count"] = float(len(positions))
            feats.update(_summarize(norm_pos, f"{name}_tercile", 3))
            feats.update(_summarize(norm_pos, f"{name}_quartile", 4))

        return feats
