"""Reduced-alphabet k-mer composition descriptor."""

from __future__ import annotations

import math
import warnings
from collections import Counter

from roxy.core.constants import REDUCED_ALPHABETS
from roxy.descriptors._utils import clean_sequence, generate_all_kmers
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_LARGE_K_THRESHOLD = 4


@register("reduced_kmer", family="composition")
class ReducedKmerDescriptor(BaseDescriptor):
    r"""K-mer frequencies over a reduced amino-acid alphabet.

    Each residue is mapped to a reduced symbol before k-mer extraction.
    Residues not in the mapping are silently dropped. Frequencies are
    ``count(kmer) / total_kmers``.

    Args:
        mapping: Dict mapping one-letter AA codes to reduced-alphabet
            symbols. Mutually exclusive with ``scheme``.
        scheme: Name of a built-in scheme from ``REDUCED_ALPHABETS``
            (``"rd5"`` or ``"rd3"``). Default ``"rd5"``.
        k: K-mer length (default 2). ``k >= 4`` emits a warning.

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``reduced_kmer_``.
        ``length``, ``alphabet_size``,
        ``freq_<kmer>`` x :math:`|\text{alphabet}|^k`.

    """

    def __init__(
        self,
        *,
        mapping: dict[str, str] | None = None,
        scheme: str = "rd5",
        k: int = 2,
    ) -> None:
        """Initialize ReducedKmerDescriptor."""
        if mapping is not None and scheme != "rd5":
            msg = "Provide either 'mapping' or 'scheme', not both."
            raise ValueError(msg)
        self.mapping: dict[str, str] = mapping if mapping is not None else REDUCED_ALPHABETS[scheme]
        self.scheme = scheme
        self.k = k
        self._alphabet: list[str] = sorted(set(self.mapping.values()))
        if k >= _LARGE_K_THRESHOLD:
            warnings.warn(
                f"ReducedKmerDescriptor with k={k} and alphabet size {len(self._alphabet)}"
                f" will produce {len(self._alphabet) ** k} columns.",
                UserWarning,
                stacklevel=2,
            )
        self._all_kmers: list[str] = generate_all_kmers(self._alphabet, k)

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute reduced k-mer features for a single sequence."""
        seq = clean_sequence(sequence)
        reduced = "".join(self.mapping.get(aa, "") for aa in seq)
        n = len(reduced)
        observed = [reduced[i : i + self.k] for i in range(n - self.k + 1)] if n >= self.k else []
        total = len(observed)
        empty = total == 0
        counts = Counter(observed)

        feats: dict[str, float] = {
            "length": float(n),
            "alphabet_size": float(len(self._alphabet)),
        }
        for km in self._all_kmers:
            feats[f"freq_{km}"] = counts.get(km, 0) / total if not empty else _NAN
        return feats
