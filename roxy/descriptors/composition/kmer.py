"""K-mer full-alphabet composition descriptor."""

from __future__ import annotations

import math
import warnings
from collections import Counter

from roxy.core.constants import AA20
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

from ._utils import clean_sequence, generate_all_kmers

_NAN = math.nan
_LARGE_K_THRESHOLD = 4


@register("kmer", family="composition")
class KmerDescriptor(BaseDescriptor):
    """K-mer counts and frequencies over the full 20-AA alphabet.

    Args:
        k: K-mer length (default 2). k >= 4 emits a warning (16000+ columns).
        include_counts: Include per-k-mer raw counts.
        include_frequencies: Include per-k-mer relative frequencies.

    Output columns (prefix ``kmer_``):
        ``length``, ``total_kmers``, ``unique_kmers``,
        optionally ``count_<kmer>`` x 20**k and ``freq_<kmer>`` x 20**k,
        ``frequency_sum``.

    """

    def __init__(
        self,
        *,
        k: int = 2,
        include_counts: bool = True,
        include_frequencies: bool = True,
    ) -> None:
        """Initialize KmerDescriptor."""
        if k >= _LARGE_K_THRESHOLD:
            warnings.warn(
                f"KmerDescriptor with k={k} will produce {20**k} feature columns"
                f" (x2 with counts+frequencies). Consider k <= 3.",
                UserWarning,
                stacklevel=2,
            )
        self.k = k
        self.include_counts = include_counts
        self.include_frequencies = include_frequencies
        self._all_kmers: list[str] = generate_all_kmers(AA20, k)

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute k-mer composition features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)
        observed = [seq[i : i + self.k] for i in range(n - self.k + 1)] if n >= self.k else []
        total = len(observed)
        empty = total == 0
        counts = Counter(observed)

        feats: dict[str, float] = {
            "length": float(n),
            "total_kmers": float(total),
            "unique_kmers": float(len(set(observed))),
        }
        if self.include_counts:
            for km in self._all_kmers:
                feats[f"count_{km}"] = float(counts.get(km, 0))
        if self.include_frequencies:
            for km in self._all_kmers:
                feats[f"freq_{km}"] = counts.get(km, 0) / total if not empty else _NAN
            freq_total = sum(counts.get(km, 0) for km in self._all_kmers)
            feats["frequency_sum"] = freq_total / total if not empty else _NAN
        return feats
