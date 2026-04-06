"""Amino-acid composition descriptor block."""

from __future__ import annotations

from typing import Iterable

from roxy.core.constants import CANONICAL_AMINO_ACID_ORDER
from roxy.sequence.descriptors._base import (
    DescriptorBlock,
    DescriptorRecord,
    prepare_descriptor_sequence,
    safe_divide,
)

CANONICAL_AA_ORDER: tuple[str, ...] = CANONICAL_AMINO_ACID_ORDER


def canonical_amino_acid_order() -> tuple[str, ...]:
    """Return the canonical amino-acid order used by AAC descriptors."""
    return CANONICAL_AA_ORDER


def _canonical_counts(
    sequence: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
) -> dict[str, int]:
    """Return canonical amino-acid counts in deterministic order."""
    cleaned = prepare_descriptor_sequence(sequence)
    ordered_alphabet = tuple(alphabet)
    return {aa: cleaned.count(aa) for aa in ordered_alphabet}


def amino_acid_counts(
    sequence: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aac_count",
) -> dict[str, int]:
    """Return absolute amino-acid counts in a reproducible order."""
    counts = _canonical_counts(sequence, alphabet=alphabet)
    return {f"{prefix}_{aa}": count for aa, count in counts.items()}


def amino_acid_frequencies(
    sequence: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aac",
) -> dict[str, float]:
    """Return normalized amino-acid frequencies in a reproducible order."""
    counts = _canonical_counts(sequence, alphabet=alphabet)
    total = sum(counts.values())
    if total == 0:
        return {f"{prefix}_{aa}": 0.0 for aa in counts}
    return {
        f"{prefix}_{aa}": safe_divide(count, total)
        for aa, count in counts.items()
    }


def amino_acid_fractions(
    sequence: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aa_frac",
) -> dict[str, float]:
    """Return amino-acid fractions with the requested public prefix."""
    return amino_acid_frequencies(
        sequence,
        alphabet=alphabet,
        prefix=prefix,
    )


class AminoAcidCompositionDescriptors(DescriptorBlock):
    """Compute amino-acid composition features in a stable canonical order."""

    family_name = "aac"
    column_prefix = "aac"

    def __init__(self, *, include_counts: bool = False) -> None:
        self.include_counts = include_counts
        super().__init__()

    def feature_names(self) -> tuple[str, ...]:
        """Return the stable AAC feature order."""
        names = tuple(f"aac_{aa}" for aa in CANONICAL_AA_ORDER)
        if not self.include_counts:
            return names
        return names + tuple(f"aac_count_{aa}" for aa in CANONICAL_AA_ORDER)

    def _transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Compute AAC frequencies and optional absolute counts."""
        features: DescriptorRecord = dict(amino_acid_frequencies(sequence))
        if self.include_counts:
            features.update(amino_acid_counts(sequence))
        return features


def aac_descriptors(
    sequence: str,
    *,
    include_counts: bool = False,
) -> DescriptorRecord:
    """Return the AAC descriptor record for one sequence."""
    block = AminoAcidCompositionDescriptors(include_counts=include_counts)
    return block.transform_sequence(sequence)


__all__ = [
    "AminoAcidCompositionDescriptors",
    "CANONICAL_AA_ORDER",
    "aac_descriptors",
    "amino_acid_counts",
    "amino_acid_fractions",
    "amino_acid_frequencies",
    "canonical_amino_acid_order",
]
