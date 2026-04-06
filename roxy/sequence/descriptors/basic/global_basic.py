"""Basic global sequence descriptor block."""

from __future__ import annotations

from typing import Iterable

from roxy.core.constants import (
    AA20,
    ALIPHATIC,
    AROMATIC,
    CF_HELIX,
    CF_SHEET,
    KD,
    NEGATIVE,
    NONPOLAR,
    PKA_C_TERM,
    PKA_N_TERM,
    PKA_SIDE,
    POLAR,
    POLARITY,
    POSITIVE,
    RESIDUE_MASS,
)
from roxy.sequence.descriptors._base import (
    DescriptorBlock,
    DescriptorRecord,
    count_residues,
    mean_scale,
    prepare_descriptor_sequence,
    safe_divide,
)
from roxy.sequence.descriptors.complexity.complexity import (
    linguistic_complexity,
    shannon_entropy,
)


def _unique_valid_residue_count(
    sequence: str,
    *,
    alphabet: Iterable[str] = AA20,
) -> int:
    """Return the number of unique valid residues in a sequence."""
    cleaned = prepare_descriptor_sequence(sequence)
    allowed = set(alphabet)
    return len({residue for residue in cleaned if residue in allowed})


def _fraction_of(
    sequence: str,
    residues: Iterable[str],
) -> float:
    """Return the residue fraction over cleaned-sequence length."""
    cleaned = prepare_descriptor_sequence(sequence)
    length = len(cleaned)
    if length == 0:
        return 0.0
    return safe_divide(count_residues(cleaned, residues), length)


def _approximate_molecular_weight(sequence: str) -> float:
    """Return an approximate molecular weight in Daltons."""
    cleaned = prepare_descriptor_sequence(sequence)
    if not cleaned:
        return 0.0
    return float(sum(RESIDUE_MASS.get(residue, 0.0) for residue in cleaned))


def _estimate_net_charge_ph7(
    sequence: str,
    *,
    include_histidine: bool = False,
) -> float:
    """Estimate net charge at pH 7.0 with a simple Henderson-Hasselbalch model."""
    cleaned = prepare_descriptor_sequence(sequence)
    if not cleaned:
        return 0.0

    pH = 7.0
    n_lys = cleaned.count("K")
    n_arg = cleaned.count("R")
    n_his = cleaned.count("H")
    n_asp = cleaned.count("D")
    n_glu = cleaned.count("E")
    n_cys = cleaned.count("C")
    n_tyr = cleaned.count("Y")

    n_term = 1.0 / (1.0 + 10.0 ** (pH - PKA_N_TERM))
    c_term = 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - pH))

    lys_charge = n_lys * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["K"])))
    arg_charge = n_arg * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["R"])))
    his_charge = (
        n_his * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["H"])))
        if include_histidine
        else 0.0
    )
    asp_charge = n_asp * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["D"] - pH)))
    glu_charge = n_glu * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["E"] - pH)))
    cys_charge = n_cys * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["C"] - pH)))
    tyr_charge = n_tyr * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["Y"] - pH)))

    positive = n_term + lys_charge + arg_charge + his_charge
    negative = c_term + asp_charge + glu_charge + cys_charge + tyr_charge
    return float(positive - negative)


class BasicGlobalDescriptors(DescriptorBlock):
    """Compute a stable P0 set of coarse global sequence descriptors."""

    family_name = "global_basic"
    column_prefix = "basic"

    def __init__(self, *, include_histidine_in_charge: bool = False) -> None:
        self.include_histidine_in_charge = include_histidine_in_charge
        super().__init__()

    def feature_names(self) -> tuple[str, ...]:
        """Return the stable feature order for the basic global block."""
        return (
            "basic_length",
            "basic_valid_residue_count",
            "basic_unique_residue_count",
            "basic_molecular_weight",
            "basic_aromatic_fraction",
            "basic_aliphatic_fraction",
            "basic_polar_fraction",
            "basic_nonpolar_fraction",
            "basic_positive_fraction",
            "basic_negative_fraction",
            "basic_charged_fraction",
            "basic_hydropathy_mean",
            "basic_polarity_mean",
            "basic_helix_propensity_mean",
            "basic_sheet_propensity_mean",
            "basic_net_charge_ph7",
            "basic_fcr",
            "basic_ncpr",
            "basic_shannon_entropy",
            "basic_linguistic_complexity_k1",
            "basic_linguistic_complexity_k2",
        )

    def _transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Compute the descriptor record for one cleaned protein sequence."""
        cleaned = prepare_descriptor_sequence(sequence)
        length = len(cleaned)
        valid_count = count_residues(cleaned, AA20)
        unique_count = _unique_valid_residue_count(cleaned, alphabet=AA20)
        net_charge = _estimate_net_charge_ph7(
            cleaned,
            include_histidine=self.include_histidine_in_charge,
        )
        complexity = linguistic_complexity(cleaned, max_k=2, alphabet_size=20)

        return {
            "length": length,
            "valid_residue_count": valid_count,
            "unique_residue_count": unique_count,
            "molecular_weight": _approximate_molecular_weight(cleaned),
            "aromatic_fraction": _fraction_of(cleaned, AROMATIC),
            "aliphatic_fraction": _fraction_of(cleaned, ALIPHATIC),
            "polar_fraction": _fraction_of(cleaned, POLAR),
            "nonpolar_fraction": _fraction_of(cleaned, NONPOLAR),
            "positive_fraction": _fraction_of(cleaned, POSITIVE),
            "negative_fraction": _fraction_of(cleaned, NEGATIVE),
            "charged_fraction": _fraction_of(cleaned, POSITIVE | NEGATIVE),
            "hydropathy_mean": mean_scale(cleaned, KD),
            "polarity_mean": mean_scale(cleaned, POLARITY),
            "helix_propensity_mean": mean_scale(cleaned, CF_HELIX),
            "sheet_propensity_mean": mean_scale(cleaned, CF_SHEET),
            "net_charge_ph7": net_charge,
            "fcr": _fraction_of(cleaned, POSITIVE | NEGATIVE),
            "ncpr": safe_divide(net_charge, length),
            "shannon_entropy": shannon_entropy(cleaned, AA20),
            "linguistic_complexity_k1": complexity["lc_k1"],
            "linguistic_complexity_k2": complexity["lc_k2"],
        }


def global_basic_descriptors(
    sequence: str,
    *,
    include_histidine_in_charge: bool = False,
) -> DescriptorRecord:
    """Return the P0 basic global descriptor record for one sequence."""
    block = BasicGlobalDescriptors(
        include_histidine_in_charge=include_histidine_in_charge,
    )
    return block.transform_sequence(sequence)


__all__ = [
    "BasicGlobalDescriptors",
    "global_basic_descriptors",
]
