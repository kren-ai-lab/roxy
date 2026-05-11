"""Functional residue content descriptors."""

from __future__ import annotations

import math

from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan

_FUNCTIONAL_GROUPS: dict[str, frozenset[str]] = {
    "catalytic_core_like":           frozenset("HSCDEKRY"),
    "catalytic_nucleophilic":        frozenset("SCYTK"),
    "acid_base_active":              frozenset("HDEKRY"),
    "redox_sensitive":               frozenset("CMYH"),
    "sulfur_containing":             frozenset("CM"),
    "aromatic_pi":                   frozenset("FWYH"),
    "hydroxyl_bearing":              frozenset("STY"),
    "amide_containing":              frozenset("NQ"),
    "flexibility_related":           frozenset("GP"),
    "basic_functional":              frozenset("KRH"),
    "acidic_functional":             frozenset("DE"),
    "phosphorylation_prone_proxy":   frozenset("STY"),
    "metal_binding_like":            frozenset("HCDE"),
    "nucleic_acid_binding_like":     frozenset("KRH"),
    "interface_like_aromatic_basic": frozenset("FWYHKR"),
    "small_reactive":                frozenset("GACS"),
}

_AA_SINGLETS = tuple("HCSDEKRYWGTPNQM")

_HBOND_DONORS: dict[str, int] = {
    "A": 0, "C": 0, "D": 0, "E": 0, "F": 0,
    "G": 0, "H": 1, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 1, "P": 0, "Q": 1, "R": 1,
    "S": 1, "T": 1, "V": 0, "W": 1, "Y": 1,
}

_HBOND_ACCEPTORS: dict[str, int] = {
    "A": 0, "C": 1, "D": 2, "E": 2, "F": 0,
    "G": 0, "H": 1, "I": 0, "K": 0, "L": 0,
    "M": 1, "N": 1, "P": 0, "Q": 1, "R": 0,
    "S": 1, "T": 1, "V": 0, "W": 0, "Y": 1,
}

# composite fractions
_COMPOSITES: dict[str, frozenset[str]] = {
    "his_cys_ser_fraction":               frozenset("HCS"),
    "asp_glu_his_fraction":               frozenset("DEH"),
    "lys_arg_his_fraction":               frozenset("KRH"),
    "triad_like_fraction":                frozenset("HSD"),
    "redox_phospho_overlap_fraction":     frozenset("CYT"),
    "aromatic_basic_fraction":            frozenset("FWYHKR"),
    "gly_pro_fraction":                   frozenset("GP"),
}


def _count(seq: str, group: frozenset[str]) -> int:
    return sum(aa in group for aa in seq)


def _frac(seq: str, group: frozenset[str], n: int) -> float:
    return _count(seq, group) / n


def _ratio(a: int, b: int) -> float:
    return a / b if b != 0 else _NAN


@register("functional_residue_content", family="motif")
class FunctionalResidueContentDescriptor(BaseDescriptor):
    """Functional residue content and proxy descriptors.

    Computes count and fraction for 16 functional groups, 15 single-residue
    types, 7 composite fractions, 4 biochemical ratios, and 3 hydrogen-bond
    proxy features.

    Output columns (prefix ``functional_residue_content_``):
        ``length``, ``valid_residue_count``,
        per group (16): ``{name}_count``, ``{name}_fraction``,
        per singlet (15): ``{aa}_count``, ``{aa}_fraction``,
        7 composite fractions, 4 ratios, 3 hbond features.
        Total: 2 + 32 + 30 + 7 + 4 + 3 = 78 columns.
    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute functional residue content features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            for name in _FUNCTIONAL_GROUPS:
                feats[f"{name}_count"] = 0.0
                feats[f"{name}_fraction"] = _NAN
            for aa in _AA_SINGLETS:
                feats[f"{aa}_count"] = 0.0
                feats[f"{aa}_fraction"] = _NAN
            for name in _COMPOSITES:
                feats[name] = _NAN
            feats["basic_acidic_ratio"] = _NAN
            feats["aromatic_sulfur_ratio"] = _NAN
            feats["hydroxyl_amide_ratio"] = _NAN
            feats["cys_met_ratio"] = _NAN
            feats["hbond_donors_per_residue"] = _NAN
            feats["hbond_acceptors_per_residue"] = _NAN
            feats["hbond_balance"] = _NAN
            return feats

        for name, group in _FUNCTIONAL_GROUPS.items():
            cnt = _count(seq, group)
            feats[f"{name}_count"] = float(cnt)
            feats[f"{name}_fraction"] = cnt / n

        for aa in _AA_SINGLETS:
            cnt = seq.count(aa)
            feats[f"{aa}_count"] = float(cnt)
            feats[f"{aa}_fraction"] = cnt / n

        for name, group in _COMPOSITES.items():
            feats[name] = _count(seq, group) / n

        feats["basic_acidic_ratio"] = _ratio(_count(seq, frozenset("KRH")), _count(seq, frozenset("DE")))
        feats["aromatic_sulfur_ratio"] = _ratio(_count(seq, frozenset("FWYH")), _count(seq, frozenset("CM")))
        feats["hydroxyl_amide_ratio"] = _ratio(_count(seq, frozenset("STY")), _count(seq, frozenset("NQ")))
        feats["cys_met_ratio"] = _ratio(seq.count("C"), seq.count("M"))

        donors = sum(_HBOND_DONORS[aa] for aa in seq) / n
        acceptors = sum(_HBOND_ACCEPTORS[aa] for aa in seq) / n
        feats["hbond_donors_per_residue"] = donors
        feats["hbond_acceptors_per_residue"] = acceptors
        feats["hbond_balance"] = donors - acceptors

        return feats
