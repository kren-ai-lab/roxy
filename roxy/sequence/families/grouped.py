"""Grouped residue composition helpers for protein sequences.

This module centralizes residue-group definitions and grouped-composition
helpers used by sequence descriptor families.

The current migration keeps the active output contract conservative:
only grouped fractions already used by the current global descriptor
implementation are emitted by default. Additional groups are defined only
when they are already clearly encoded by existing constants.

Edge-case policy
----------------
- Empty cleaned sequences return zero-valued grouped fractions.
- Length-1 sequences are valid and produce deterministic single-residue
  fractions.
- Unknown residues are handled by the calling cleaning policy. Direct
  low-level calls remain permissive and count only residues present in
  the requested groups.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

from roxy.core.constants import (
    AROMATIC,
    HYDROPHOBIC,
    NEGATIVE,
    NONPOLAR,
    POLAR,
    POLAR_UNCHARGED,
    POSITIVE,
)

from roxy.sequence.utils import clean_sequence, count_residues, safe_divide

# NOTE:
# - Keep these definitions explicit and close to the grouped descriptor
#   helpers so future sequence families can reuse them consistently.
# - Only groups already present in the current code path or directly
#   encoded by existing constants are defined here for now.
GROUPED_RESIDUE_SETS: Dict[str, frozenset[str]] = {
    "aromatic": frozenset(AROMATIC),
    "positive": frozenset(POSITIVE),
    "negative": frozenset(NEGATIVE),
    "charged": frozenset(POSITIVE | NEGATIVE),
    "polar": frozenset(POLAR),
    "polar_uncharged": frozenset(POLAR_UNCHARGED),
    "nonpolar": frozenset(NONPOLAR),
    "hydrophobic": frozenset(HYDROPHOBIC),
    "hydrophilic": frozenset(POLAR),
}

# NOTE:
# - These are the grouped fraction outputs currently used by the migrated
#   global/basic descriptor block. Keep names stable while the monolith
#   still delegates to this module.
DEFAULT_GROUPED_FRACTION_GROUPS: Dict[str, frozenset[str]] = {
    "frac_aromatic": GROUPED_RESIDUE_SETS["aromatic"],
    "frac_positive": GROUPED_RESIDUE_SETS["positive"],
    "frac_negative": GROUPED_RESIDUE_SETS["negative"],
    "frac_polar": GROUPED_RESIDUE_SETS["polar"],
    "frac_nonpolar": GROUPED_RESIDUE_SETS["nonpolar"],
}


def grouped_residue_fractions(
    seq: str,
    *,
    groups: Mapping[str, Iterable[str]],
) -> Dict[str, float]:
    """Compute per-group fractions over a cleaned protein sequence.

    Empty cleaned sequences yield all-zero grouped fractions.
    """
    s = clean_sequence(seq)
    length = len(s)
    if length == 0:
        return {name: 0.0 for name in groups}

    return {
        name: safe_divide(count_residues(s, residues), length)
        for name, residues in groups.items()
    }


def grouped_fraction_descriptors(seq: str) -> Dict[str, float]:
    """Return the grouped fraction block used by current global summaries."""
    return grouped_residue_fractions(
        seq,
        groups=DEFAULT_GROUPED_FRACTION_GROUPS,
    )


# TODO:
# - Add richer grouped families such as aliphatic, tiny, small, branched,
#   sulfur-containing, hydroxyl-containing, amide-containing,
#   disorder-promoting, and order-promoting only after those groups are
#   formally specified in active constants or migrated descriptor code.
# - Add ratio/balance descriptors in a separate helper layer once the
#   grouped-family public contract is defined.

__all__ = [
    "DEFAULT_GROUPED_FRACTION_GROUPS",
    "GROUPED_RESIDUE_SETS",
    "grouped_fraction_descriptors",
    "grouped_residue_fractions",
]
