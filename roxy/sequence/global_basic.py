"""Global basic protein sequence descriptor helpers.

This module hosts coarse-grained, sequence-wide descriptor logic that is
shared across global sequence descriptor families.

The helpers here intentionally remain simple and side-effect free so
that sequence-level APIs can reuse them consistently across descriptor
families.

Edge-case policy
----------------
- Empty cleaned sequences do not raise at the descriptor level. They
  return the canonical default block from
  :func:`empty_global_descriptor_defaults`.
- Length-1 sequences are valid and produce deterministic scalar
  summaries.
- Unknown residues are governed by the public cleaning policy before
  reaching this module. Direct low-level calls remain permissive and
  ignore unknown residues in scale- or class-based summaries.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

from roxy.core.constants import (
    AA20,
    ACCEPTORS,
    BOMAN,
    CF_HELIX,
    CF_SHEET,
    DONORS,
    EISENBERG,
    KD,
    NEGATIVE,
    PKA_C_TERM,
    PKA_N_TERM,
    PKA_SIDE,
    POSITIVE,
    TOP_IDP,
)

from .grouped import grouped_fraction_descriptors
from .utils import (
    clean_sequence,
    count_residues,
    linguistic_complexity,
    mean_scale,
    safe_divide,
    shannon_entropy,
    sum_residue_values,
)


def sequence_length(seq: str) -> int:
    """Return the cleaned-sequence length used by current descriptors."""
    return len(clean_sequence(seq))


def valid_residue_count(
    seq: str,
    *,
    alphabet: Iterable[str] = AA20,
) -> int:
    """Count canonical residues present in a cleaned sequence."""
    return count_residues(seq, alphabet)


def unique_residue_count(
    seq: str,
    *,
    alphabet: Iterable[str] = AA20,
) -> int:
    """Count the number of unique canonical residues in a cleaned sequence."""
    s = clean_sequence(seq)
    allowed = set(alphabet)
    return len({residue for residue in s if residue in allowed})


def estimate_net_charge(
    seq: str,
    *,
    pH: float = 7.0,
    include_histidine: bool = False,
) -> float:
    """Estimate net charge using a Henderson-Hasselbalch approximation."""
    s = clean_sequence(seq)
    if not s:
        return 0.0

    nK = s.count("K")
    nR = s.count("R")
    nH = s.count("H")
    nD = s.count("D")
    nE = s.count("E")
    nC = s.count("C")
    nY = s.count("Y")

    nterm = 1.0 / (1.0 + 10.0 ** (pH - PKA_N_TERM))
    cterm = 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - pH))

    charge_k = nK * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["K"])))
    charge_r = nR * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["R"])))
    charge_h = (
        nH * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["H"])))
        if include_histidine
        else 0.0
    )
    charge_d = nD * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["D"] - pH)))
    charge_e = nE * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["E"] - pH)))
    charge_c = nC * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["C"] - pH)))
    charge_y = nY * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["Y"] - pH)))

    positive = nterm + charge_k + charge_r + charge_h
    negative = cterm + charge_d + charge_e + charge_c + charge_y
    return float(positive - negative)


def scale_summary_descriptors(seq: str) -> Dict[str, float]:
    """Return the scale-based global summaries currently used by Roxy."""
    s = clean_sequence(seq)
    return {
        "gravy_kd": mean_scale(s, KD),
        "hydropathy_eisenberg": mean_scale(s, EISENBERG),
        "top_idp_mean": mean_scale(s, TOP_IDP),
        "helix_propensity_mean": mean_scale(s, CF_HELIX),
        "sheet_propensity_mean": mean_scale(s, CF_SHEET),
    }


def charge_descriptors(
    seq: str,
    *,
    pH: float = 7.0,
    include_histidine: bool = False,
) -> Dict[str, float]:
    """Return sequence-level charge summaries."""
    s = clean_sequence(seq)
    length = sequence_length(s)
    net_charge = estimate_net_charge(
        s,
        pH=pH,
        include_histidine=include_histidine,
    )
    n_pos = count_residues(s, POSITIVE)
    n_neg = count_residues(s, NEGATIVE)
    return {
        "net_charge_pH": net_charge,
        "fcr": safe_divide(n_pos + n_neg, length),
        "ncpr": safe_divide(net_charge, length),
    }


def donor_acceptor_descriptors(seq: str) -> Dict[str, float]:
    """Return donor/acceptor summaries normalized by cleaned length."""
    s = clean_sequence(seq)
    length = sequence_length(s)
    n_donors = sum_residue_values(s, DONORS)
    n_acceptors = sum_residue_values(s, ACCEPTORS)
    return {
        "donors_per_residue": safe_divide(n_donors, length),
        "acceptors_per_residue": safe_divide(n_acceptors, length),
    }


def entropy_complexity_descriptors(
    seq: str,
    *,
    max_k: int = 3,
) -> Dict[str, float]:
    """Return entropy and simple linguistic-complexity summaries."""
    s = clean_sequence(seq)
    feats = {"aa_entropy": shannon_entropy(s, AA20)}
    feats.update(linguistic_complexity(s, max_k=max_k, alphabet_size=20))
    return feats


def boman_index(seq: str) -> float:
    """Return the sequence-level Boman index used by current summaries."""
    s = clean_sequence(seq)
    return safe_divide(
        sum_residue_values(s, BOMAN),
        sequence_length(s),
        default=math.nan,
    )


def global_basic_descriptors(
    seq: str,
    *,
    pH: float = 7.0,
    include_histidine_in_charge: bool = False,
) -> Dict[str, float]:
    """Assemble the current non-composition global-basic descriptor block.

    Empty cleaned sequences return the canonical default block. This
    keeps descriptor outputs deterministic even when higher-level API
    validation allows empty inputs.
    """
    s = clean_sequence(seq)
    length = sequence_length(s)
    if length == 0:
        return empty_global_descriptor_defaults()

    feats = grouped_fraction_descriptors(s)
    feats.update(scale_summary_descriptors(s))
    feats["boman_index"] = boman_index(s)
    feats.update(
        charge_descriptors(
            s,
            pH=pH,
            include_histidine=include_histidine_in_charge,
        )
    )
    feats.update(donor_acceptor_descriptors(s))
    feats.update(entropy_complexity_descriptors(s, max_k=3))
    return feats


def empty_global_descriptor_defaults(
    *,
    aaindex_codes: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Return the canonical empty-sequence defaults used by Roxy."""
    feats: Dict[str, float] = {
        "frac_aromatic": 0.0,
        "frac_positive": 0.0,
        "frac_negative": 0.0,
        "frac_polar": 0.0,
        "frac_nonpolar": 0.0,
        "gravy_kd": math.nan,
        "hydropathy_eisenberg": math.nan,
        "top_idp_mean": math.nan,
        "helix_propensity_mean": math.nan,
        "sheet_propensity_mean": math.nan,
        "boman_index": math.nan,
        "net_charge_pH": 0.0,
        "fcr": 0.0,
        "ncpr": 0.0,
        "donors_per_residue": 0.0,
        "acceptors_per_residue": 0.0,
        "aa_entropy": math.nan,
        "lc_k1": 0.0,
        "lc_k2": 0.0,
        "lc_k3": 0.0,
    }

    if aaindex_codes is not None:
        for code in aaindex_codes:
            feats[f"aaindex_{code}_mean"] = math.nan

    return feats


# TODO:
# - Add molecular-weight approximations only when they are explicitly
#   introduced in the active descriptor contract.
# - Add aliphatic or other derived indices only when they exist in the
#   previous implementation or are formally specified.

__all__ = [
    "boman_index",
    "charge_descriptors",
    "donor_acceptor_descriptors",
    "empty_global_descriptor_defaults",
    "entropy_complexity_descriptors",
    "estimate_net_charge",
    "global_basic_descriptors",
    "scale_summary_descriptors",
    "sequence_length",
    "unique_residue_count",
    "valid_residue_count",
]
