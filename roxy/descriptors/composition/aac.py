"""Amino acid composition (AAC) descriptor."""

from __future__ import annotations

import math
from collections import Counter

from roxy.core.constants import AA20_ORDERED
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan


@register("aac", family="composition")
class AACDescriptor(BaseDescriptor):
    """Amino acid counts and frequencies over the 20 standard residues.

    Args:
        include_counts: Include per-residue raw counts.
        include_frequencies: Include per-residue relative frequencies.

    Output columns (prefix ``aac_``):
        ``length``, ``valid_residue_count``, ``unique_residue_count``,
        optionally ``count_<AA>`` x 20 and ``freq_<AA>`` x 20,
        ``frequency_sum``.

    """

    def __init__(
        self,
        *,
        include_counts: bool = True,
        include_frequencies: bool = True,
    ) -> None:
        """Initialize AACDescriptor."""
        self.include_counts = include_counts
        self.include_frequencies = include_frequencies

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute AAC features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)
        empty = n == 0
        counts = Counter(seq)

        feats["unique_residue_count"] = float(len(set(seq))) if not empty else 0.0
        if self.include_counts:
            for aa in AA20_ORDERED:
                feats[f"count_{aa}"] = float(counts.get(aa, 0))
        if self.include_frequencies:
            for aa in AA20_ORDERED:
                feats[f"freq_{aa}"] = counts.get(aa, 0) / n if not empty else _NAN
            feats["frequency_sum"] = sum(counts.get(aa, 0) for aa in AA20_ORDERED) / n if not empty else _NAN
        return feats
