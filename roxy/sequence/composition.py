"""Composition-based sequence descriptors.

This module is the dedicated home for amino-acid composition (AAC)
helpers in the new sequence-focused architecture.

The first extraction goal is conservative: keep the previous monolithic
descriptor behavior stable while introducing a reproducible AAC building
block that other sequence families can reuse.

Edge-case policy
----------------
- Empty cleaned sequences return zero-valued AAC outputs.
- Length-1 sequences are valid and produce a single residue frequency of
  ``1.0`` with the remaining AAC features equal to ``0.0``.
- Unknown residues are handled by the calling cleaning policy. Direct
  low-level calls remain permissive and count only matching residues in
  the requested alphabet.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

from .utils import clean_sequence, safe_divide

# NOTE:
# - Use a fixed canonical order rather than `AA20` from core constants,
#   because `AA20` is a set and therefore does not guarantee stable
#   output-column ordering across callers.
CANONICAL_AA_ORDER: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")


def canonical_amino_acid_order() -> Tuple[str, ...]:
    """Return the canonical amino-acid order used by AAC helpers."""
    return CANONICAL_AA_ORDER


def amino_acid_counts(
    seq: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aac_count",
) -> Dict[str, int]:
    """Return absolute amino-acid counts in a reproducible order."""
    s = clean_sequence(seq)
    ordered_alphabet = tuple(alphabet)
    return {
        f"{prefix}_{aa}": s.count(aa)
        for aa in ordered_alphabet
    }


def amino_acid_frequencies(
    seq: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aac",
) -> Dict[str, float]:
    """Return normalized amino-acid frequencies in a reproducible order.

    Empty cleaned sequences yield all-zero AAC values rather than
    raising an error.
    """
    s = clean_sequence(seq)
    ordered_alphabet = tuple(alphabet)
    length = len(s)
    if length == 0:
        return {f"{prefix}_{aa}": 0.0 for aa in ordered_alphabet}
    return {
        f"{prefix}_{aa}": safe_divide(s.count(aa), length)
        for aa in ordered_alphabet
    }


def amino_acid_fractions(
    seq: str,
    *,
    alphabet: Iterable[str] = CANONICAL_AA_ORDER,
    prefix: str = "aa_frac",
) -> Dict[str, float]:
    """Return amino-acid fractions with the current public feature names.

    This helper preserves the current feature names used by
    the sequence descriptor API while delegating to the AAC core.
    """
    return amino_acid_frequencies(
        seq,
        alphabet=alphabet,
        prefix=prefix,
    )


def aac_descriptors(
    seq: str,
    *,
    include_counts: bool = False,
) -> Dict[str, float]:
    """Return the new AAC descriptor block for future sequence APIs."""
    feats: Dict[str, float] = dict(amino_acid_frequencies(seq))
    if include_counts:
        feats.update(amino_acid_counts(seq))
    return feats


# TODO:
# - Add optional count-based AAC outputs to the public sequence API only
#   after the active descriptor contract explicitly includes them.
# - Evaluate whether entropy-derived AAC summaries should live here or in
#   `roxy.sequence.global_basic` once the public API is stabilized.

__all__ = [
    "CANONICAL_AA_ORDER",
    "aac_descriptors",
    "amino_acid_counts",
    "amino_acid_fractions",
    "amino_acid_frequencies",
    "canonical_amino_acid_order",
]
