"""Dipeptide composition descriptors."""

from __future__ import annotations

from typing import Dict, Tuple

from roxy.sequence.descriptors._base import DescriptorBlock
from roxy.sequence.descriptors.basic.composition import canonical_amino_acid_order
from roxy.sequence.descriptors.kmers.full import (
    KMerDescriptors,
    count_kmers,
    kmer_frequencies,
    normalize_kmer_sequence,
    ordered_kmer_space,
)


def dpc_descriptors(
    sequence: str,
    *,
    include_counts: bool = True,
    include_frequencies: bool = True,
) -> Dict[str, float]:
    """Return notebook-style DPC descriptors on the canonical alphabet."""
    cleaned = normalize_kmer_sequence(sequence)
    dipeptide_counts = count_kmers(
        cleaned,
        2,
        alphabet=canonical_amino_acid_order(),
    )
    total_dipeptides = int(sum(dipeptide_counts.values()))
    unique_dipeptides = int(sum(1 for count in dipeptide_counts.values() if count > 0))

    features: Dict[str, float] = {
        "dpc_length": float(len(cleaned)),
        "dpc_valid_residue_count": float(len(cleaned)),
        "dpc_total_dipeptides": float(total_dipeptides),
        "dpc_unique_dipeptides": float(unique_dipeptides),
    }

    if include_counts:
        features.update(
            {
                f"dpc_count_{key}": float(value)
                for key, value in dipeptide_counts.items()
            }
        )
    if include_frequencies:
        features.update(
            kmer_frequencies(
                cleaned,
                2,
                alphabet=canonical_amino_acid_order(),
                prefix="dpc_freq",
            )
        )
        features["dpc_frequency_sum"] = float(
            sum(
                value
                for key, value in features.items()
                if key.startswith("dpc_freq_")
            )
        )
    else:
        features["dpc_frequency_sum"] = 0.0

    return features


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


class DipeptideDescriptors(KMerDescriptors):
    """Notebook-aligned DPC descriptor block on the canonical AA20 alphabet."""

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
        self.family_name = "dpc"
        self.column_prefix = "dpc"
        self.include_counts = include_counts
        self.include_frequencies = include_frequencies
        DescriptorBlock.__init__(self)

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

    def _transform_sequence(self, sequence: str) -> Dict[str, float]:
        """Compute notebook-style DPC descriptors for a single sequence."""
        return dpc_descriptors(
            sequence,
            include_counts=self.include_counts,
            include_frequencies=self.include_frequencies,
        )


__all__ = [
    "DipeptideDescriptors",
    "dpc_count_descriptors",
    "dpc_descriptors",
    "dpc_frequency_descriptors",
]
