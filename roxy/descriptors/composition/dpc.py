"""Dipeptide composition (DPC) descriptor."""

from __future__ import annotations

import math
from collections import Counter

from roxy.core.constants import AA20
from roxy.descriptors._utils import generate_all_kmers
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_ALL_DIPEPTIDES: list[str] = generate_all_kmers(AA20, 2)


@register("dpc", family="composition")
class DPCDescriptor(BaseDescriptor):
    """Dipeptide counts and frequencies (400 dipeptides over standard AAs).

    Args:
        include_counts: Include per-dipeptide raw counts.
        include_frequencies: Include per-dipeptide relative frequencies.

    Output columns (prefix ``dpc_``):
        ``length``, ``valid_residue_count``, ``total_dipeptides``,
        ``unique_dipeptides``, optionally ``count_<XY>`` x 400 and
        ``freq_<XY>`` x 400, ``frequency_sum``.

    """

    def __init__(
        self,
        *,
        include_counts: bool = True,
        include_frequencies: bool = True,
    ) -> None:
        """Initialize DPCDescriptor."""
        self.include_counts = include_counts
        self.include_frequencies = include_frequencies

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute DPC features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)
        dipeptides = [seq[i : i + 2] for i in range(n - 1)] if n >= 2 else []  # noqa: PLR2004
        total = len(dipeptides)
        empty = total == 0
        counts = Counter(dipeptides)

        feats["total_dipeptides"] = float(total)
        feats["unique_dipeptides"] = float(len(set(dipeptides)))
        if self.include_counts:
            for dp in _ALL_DIPEPTIDES:
                feats[f"count_{dp}"] = float(counts.get(dp, 0))
        if self.include_frequencies:
            for dp in _ALL_DIPEPTIDES:
                feats[f"freq_{dp}"] = counts.get(dp, 0) / total if not empty else _NAN
            total_count = sum(counts.get(dp, 0) for dp in _ALL_DIPEPTIDES)
            feats["frequency_sum"] = total_count / total if not empty else _NAN
        return feats
