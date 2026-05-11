"""Grouped amino acid composition descriptor."""

from __future__ import annotations

import math

from roxy.core.constants import AA_GROUPS
from roxy.descriptors._utils import clean_sequence, safe_ratio
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan


@register("grouped", family="composition")
class GroupedCompositionDescriptor(BaseDescriptor):
    """Per-group residue counts and fractions plus four compositional ratios.

    Groups (17 total, defined in ``roxy.core.constants.AA_GROUPS``):
    positive, negative, charged, polar, nonpolar, aromatic, aliphatic,
    tiny, small, branched, sulfur, hydroxyl, amide, hydrophobic,
    hydrophilic, disorder_promoting, order_promoting.

    Output columns (prefix ``grouped_``):
        ``length``, ``<group>_count`` x 17, ``<group>_frac`` x 17,
        ``ratio_acidic_basic``, ``ratio_basic_acidic``,
        ``ratio_polar_nonpolar``, ``ratio_charged_uncharged``.

    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute grouped composition features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)
        empty = n == 0

        counts = {name: sum(aa in grp for aa in seq) for name, grp in AA_GROUPS.items()}

        feats: dict[str, float] = {"length": float(n)}
        for name, cnt in counts.items():
            feats[f"{name}_count"] = float(cnt)
            feats[f"{name}_frac"] = cnt / n if not empty else _NAN

        feats["ratio_acidic_basic"] = safe_ratio(counts["negative"], counts["positive"])
        feats["ratio_basic_acidic"] = safe_ratio(counts["positive"], counts["negative"])
        feats["ratio_polar_nonpolar"] = safe_ratio(counts["polar"], counts["nonpolar"])
        uncharged = n - counts["charged"]
        feats["ratio_charged_uncharged"] = safe_ratio(counts["charged"], uncharged)

        return feats
