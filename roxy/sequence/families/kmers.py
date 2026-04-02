"""K-mer sequence descriptors.

This module provides a small generic k-mer implementation for the active
sequence architecture and a dedicated DPC block for dipeptide
composition.

Edge-case policy
----------------
- Empty cleaned sequences return a zero-valued descriptor vector.
- Sequences shorter than ``k`` return a zero-valued descriptor vector.
- Sequence cleaning follows the active preprocessing layer and drops
  non-canonical residues for the DPC-oriented path.
- For canonical sequences with length ``>= k``, normalized k-mer
  frequencies sum to ``1.0``.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from roxy.sequence.base import BaseSequenceDescriptorBlock
from roxy.sequence.families.composition import (
    CANONICAL_AA_ORDER,
    canonical_amino_acid_order,
)
from roxy.sequence.preprocessing.cleaning import (
    SequenceCleaningConfig,
    clean_sequence as preprocess_sequence,
)
from roxy.sequence.utils import safe_divide

_KMER_CLEANING_CONFIG = SequenceCleaningConfig(invalid_policy="drop")


def ordered_kmer_space(
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AA_ORDER,
) -> Tuple[str, ...]:
    """Return the full ordered k-mer space for ``k`` and ``alphabet``.

    The ordering is deterministic and follows the cartesian-product order
    induced by the provided alphabet.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    ordered_alphabet = tuple(alphabet)
    return tuple("".join(chars) for chars in product(ordered_alphabet, repeat=k))


def _valid_kmer_windows(
    sequence: str,
    *,
    k: int,
    alphabet: Sequence[str],
) -> Tuple[str, ...]:
    """Return valid k-mer windows from a cleaned sequence.

    Windows containing symbols outside ``alphabet`` are ignored. In the
    default DPC-oriented path, non-canonical residues are dropped during
    cleaning before windows are extracted.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    s = _normalize_for_kmers(sequence)
    if len(s) < k:
        return ()

    allowed = set(alphabet)
    windows = []
    for i in range(len(s) - k + 1):
        window = s[i : i + k]
        if all(token in allowed for token in window):
            windows.append(window)
    return tuple(windows)


def _normalize_for_kmers(sequence: object) -> str:
    """Normalize a sequence with the shared preprocessing layer.

    The k-mer family uses the shared active preprocessing rules and drops
    non-canonical residues so DPC can reproduce the notebook-style
    canonical AA20 behavior.
    """
    cleaned = preprocess_sequence(sequence, config=_KMER_CLEANING_CONFIG)
    return "" if cleaned is None else cleaned


def count_kmers(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AA_ORDER,
) -> Dict[str, int]:
    """Count observed k-mers in deterministic output order.

    If the cleaned sequence is shorter than ``k``, the returned count
    vector is all zeros.
    """
    space = ordered_kmer_space(k, alphabet=alphabet)
    counts = {kmer: 0 for kmer in space}

    for kmer in _valid_kmer_windows(sequence, k=k, alphabet=alphabet):
        counts[kmer] += 1

    return counts


def kmer_frequencies(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AA_ORDER,
    prefix: str | None = None,
) -> Dict[str, float]:
    """Return normalized k-mer frequencies in deterministic order.

    Frequencies are normalized by the number of valid observed windows.
    If there are no valid windows, the result is an all-zero vector.
    """
    counts = count_kmers(sequence, k, alphabet=alphabet)
    windows = _valid_kmer_windows(sequence, k=k, alphabet=alphabet)
    total = len(windows)
    name_prefix = prefix or f"kmer{k}"

    return {
        f"{name_prefix}_{kmer}": safe_divide(count, total)
        for kmer, count in counts.items()
    }


def kmer_counts(
    sequence: str,
    k: int,
    *,
    alphabet: Sequence[str] = CANONICAL_AA_ORDER,
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
    alphabet: Sequence[str] = CANONICAL_AA_ORDER,
    include_counts: bool = False,
) -> Dict[str, float]:
    """Return general k-mer descriptors for a protein sequence."""
    feats: Dict[str, float] = dict(
        kmer_frequencies(
            sequence,
            k,
            alphabet=alphabet,
            prefix=f"kmer{k}",
        )
    )
    if include_counts:
        feats.update(
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
    return feats


def dpc_descriptors(
    sequence: str,
    *,
    include_counts: bool = True,
    include_frequencies: bool = True,
) -> Dict[str, float]:
    """Return notebook-style DPC descriptors on the canonical alphabet.

    The returned block includes summary fields plus optional count and
    frequency vectors. Frequency features use the ``dpc_freq_*`` naming
    convention so the active library can reproduce the reference DPC
    notebook more closely.
    """
    cleaned = _normalize_for_kmers(sequence)
    dipeptide_counts = count_kmers(
        cleaned,
        2,
        alphabet=canonical_amino_acid_order(),
    )
    total_dipeptides = int(sum(dipeptide_counts.values()))
    unique_dipeptides = int(sum(1 for count in dipeptide_counts.values() if count > 0))

    feats: Dict[str, float] = {
        "dpc_length": float(len(cleaned)),
        "dpc_valid_residue_count": float(len(cleaned)),
        "dpc_total_dipeptides": float(total_dipeptides),
        "dpc_unique_dipeptides": float(unique_dipeptides),
    }

    if include_counts:
        feats.update(
            {
                f"dpc_count_{key}": float(value)
                for key, value in dipeptide_counts.items()
            }
        )
    if include_frequencies:
        feats.update(
            kmer_frequencies(
                cleaned,
                2,
                alphabet=canonical_amino_acid_order(),
                prefix="dpc_freq",
            )
        )
        feats["dpc_frequency_sum"] = float(
            sum(
                value
                for key, value in feats.items()
                if key.startswith("dpc_freq_")
            )
        )
    else:
        feats["dpc_frequency_sum"] = 0.0

    return feats


def dpc_count_descriptors(sequence: str) -> Dict[str, float]:
    """Return only the notebook-style DPC count features."""
    return {
        key: value
        for key, value in dpc_descriptors(
            sequence,
            include_counts=True,
            include_frequencies=False,
        ).items()
        if key.startswith("dpc_count_")
    }


def dpc_frequency_descriptors(sequence: str) -> Dict[str, float]:
    """Return only the notebook-style DPC frequency features."""
    return {
        key: value
        for key, value in dpc_descriptors(
            sequence,
            include_counts=False,
            include_frequencies=True,
        ).items()
        if key.startswith("dpc_freq_")
    }


class KMerDescriptors(BaseSequenceDescriptorBlock):
    """Configurable k-mer descriptor block with deterministic ordering."""

    def __init__(
        self,
        k: int,
        *,
        alphabet: Sequence[str] = CANONICAL_AA_ORDER,
        include_counts: bool = False,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        self.k = int(k)
        self.alphabet = tuple(alphabet)
        self.include_counts = include_counts
        self.name = f"kmer{self.k}"

    def feature_names(self) -> Tuple[str, ...]:
        """Return the stable k-mer feature order for this block."""
        kmers = ordered_kmer_space(self.k, alphabet=self.alphabet)
        names = tuple(f"kmer{self.k}_{pattern}" for pattern in kmers)
        if not self.include_counts:
            return names
        return names + tuple(f"kmer{self.k}_count_{pattern}" for pattern in kmers)

    def describe_sequence(self, sequence: str) -> Mapping[str, float]:
        """Compute k-mer descriptors for a single sequence."""
        return kmer_descriptors(
            sequence,
            self.k,
            alphabet=self.alphabet,
            include_counts=self.include_counts,
        )


class DipeptideDescriptors(KMerDescriptors):
    """Notebook-aligned DPC descriptor block on the canonical AA20 alphabet."""

    name = "dpc"

    def __init__(
        self,
        *,
        include_counts: bool = True,
        include_frequencies: bool = True,
    ) -> None:
        super().__init__(
            2,
            alphabet=canonical_amino_acid_order(),
            include_counts=include_counts,
        )
        self.name = "dpc"
        self.include_counts = include_counts
        self.include_frequencies = include_frequencies

    def feature_names(self) -> Tuple[str, ...]:
        """Return the stable notebook-style DPC feature order."""
        kmers = ordered_kmer_space(2, alphabet=self.alphabet)
        names = [
            "dpc_length",
            "dpc_valid_residue_count",
            "dpc_total_dipeptides",
            "dpc_unique_dipeptides",
        ]
        if self.include_counts:
            names.extend(f"dpc_count_{pattern}" for pattern in kmers)
        if self.include_frequencies:
            names.extend(f"dpc_freq_{pattern}" for pattern in kmers)
        names.append("dpc_frequency_sum")
        return tuple(names)

    def describe_sequence(self, sequence: str) -> Mapping[str, float]:
        """Compute notebook-style DPC descriptors for a single sequence."""
        return dpc_descriptors(
            sequence,
            include_counts=self.include_counts,
            include_frequencies=self.include_frequencies,
        )


__all__ = [
    "DipeptideDescriptors",
    "KMerDescriptors",
    "count_kmers",
    "dpc_count_descriptors",
    "dpc_descriptors",
    "dpc_frequency_descriptors",
    "kmer_counts",
    "kmer_descriptors",
    "kmer_frequencies",
    "ordered_kmer_space",
]
