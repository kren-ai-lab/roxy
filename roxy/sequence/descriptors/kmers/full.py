"""General k-mer sequence descriptors."""

from __future__ import annotations

from itertools import product
from typing import Dict, Sequence, Tuple

from roxy.core.constants import CANONICAL_AMINO_ACID_ORDER
from roxy.sequence.cleaning import (
    SequenceCleaningConfig,
    clean_sequence as preprocess_sequence,
)
from roxy.sequence.descriptors._base import DescriptorBlock, safe_divide

_KMER_CLEANING_CONFIG = SequenceCleaningConfig(invalid_policy="drop")


def ordered_kmer_space(
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
) -> Tuple[str, ...]:
    """Return the full ordered k-mer space for ``k`` and ``alphabet``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    ordered_alphabet = tuple(alphabet)
    return tuple("".join(chars) for chars in product(ordered_alphabet, repeat=k))


def normalize_kmer_sequence(sequence: object) -> str:
    """Normalize a sequence for k-mer extraction under the shared policy."""
    cleaned = preprocess_sequence(sequence, config=_KMER_CLEANING_CONFIG)
    return "" if cleaned is None else cleaned


def _valid_kmer_windows(
    sequence: str,
    *,
    k: int,
    alphabet: Sequence[str],
) -> Tuple[str, ...]:
    """Return valid k-mer windows from a cleaned sequence."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    cleaned = normalize_kmer_sequence(sequence)
    if len(cleaned) < k:
        return ()

    allowed = set(alphabet)
    windows = []
    for i in range(len(cleaned) - k + 1):
        window = cleaned[i : i + k]
        if all(token in allowed for token in window):
            windows.append(window)
    return tuple(windows)


def count_kmers(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
) -> Dict[str, int]:
    """Count observed k-mers in deterministic output order."""
    space = ordered_kmer_space(k, alphabet=alphabet)
    counts = {kmer: 0 for kmer in space}

    for kmer in _valid_kmer_windows(sequence, k=k, alphabet=alphabet):
        counts[kmer] += 1

    return counts


def kmer_frequencies(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
    prefix: str | None = None,
) -> Dict[str, float]:
    """Return normalized k-mer frequencies in deterministic order."""
    counts = count_kmers(sequence, k, alphabet=alphabet)
    total = len(_valid_kmer_windows(sequence, k=k, alphabet=alphabet))
    name_prefix = prefix or f"kmer{k}"

    return {
        f"{name_prefix}_{kmer}": safe_divide(count, total)
        for kmer, count in counts.items()
    }


def kmer_counts(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
    prefix: str | None = None,
) -> Dict[str, int]:
    """Return k-mer counts with deterministic feature ordering."""
    counts = count_kmers(sequence, k, alphabet=alphabet)
    name_prefix = prefix or f"kmer{k}_count"
    return {f"{name_prefix}_{kmer}": count for kmer, count in counts.items()}


def kmer_descriptors(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
    include_counts: bool = False,
) -> Dict[str, float]:
    """Return general k-mer descriptors for a protein sequence."""
    features: Dict[str, float] = dict(
        kmer_frequencies(
            sequence,
            k,
            alphabet=alphabet,
            prefix=f"kmer{k}",
        )
    )
    if include_counts:
        features.update(
            {
                key: float(value)
                for key, value in kmer_counts(
                    sequence,
                    k,
                    alphabet=alphabet,
                    prefix=f"kmer{k}_count",
                ).items()
            }
        )
    return features


class KMerDescriptors(DescriptorBlock):
    """Configurable k-mer descriptor block with deterministic ordering."""

    def __init__(
        self,
        k: int,
        *,
        alphabet: Sequence[str] = CANONICAL_AMINO_ACID_ORDER,
        include_counts: bool = False,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        self.k = int(k)
        self.alphabet = tuple(alphabet)
        self.include_counts = include_counts
        self.family_name = f"kmer{self.k}"
        self.column_prefix = f"kmer{self.k}"
        super().__init__()

    def feature_names(self) -> Tuple[str, ...]:
        """Return the stable k-mer feature order for this block."""
        kmers = ordered_kmer_space(self.k, alphabet=self.alphabet)
        names = tuple(f"kmer{self.k}_{pattern}" for pattern in kmers)
        if not self.include_counts:
            return names
        return names + tuple(f"kmer{self.k}_count_{pattern}" for pattern in kmers)

    def _transform_sequence(self, sequence: str) -> Dict[str, float]:
        """Compute k-mer descriptors for a single sequence."""
        return kmer_descriptors(
            sequence,
            self.k,
            alphabet=self.alphabet,
            include_counts=self.include_counts,
        )


__all__ = [
    "KMerDescriptors",
    "count_kmers",
    "kmer_counts",
    "kmer_descriptors",
    "kmer_frequencies",
    "normalize_kmer_sequence",
    "ordered_kmer_space",
]
