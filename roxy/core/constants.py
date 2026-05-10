"""Constants and lookup tables for protein sequence descriptor calculations.

This module centralises amino-acid level constants that are reused across
the protein descriptor toolbox. It intentionally contains **no business
logic**, only data, so that it can be safely imported from different parts of
the codebase (e.g. descriptor engines, analysis utilities, etc.).

The constants defined here include:

- Column name lists used when working with antibody or chain-level datasets.
- The canonical set of 20 amino acids (`AA20`).
- Hydrophobicity scales (Kyte–Doolittle, Eisenberg).
- Boman index contributions.
- Chou–Fasman helix and sheet propensities.
- TOP-IDP intrinsic disorder scale.
- pKa values for termini and side chains.
- Residue class sets (aromatic, charged, polar, hydrophobic).
- Hydrogen-bond donor and acceptor counts.

All amino acid codes follow the one-letter, uppercase convention.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column name conventions for antibody / chain datasets
# ---------------------------------------------------------------------------

#: Metadata columns typically associated with antibody-level identifiers.
META_COLS: list[str] = ["id", "id.pasteur", "aid", "subset", "ighv"]

#: Columns commonly found in chain-level (e.g. heavy/light chain) tables.
CHAIN_COLS: list[str] = [
    "sequence",
    "junction",
    "junction_aa",
    "cdr1",
    "cdr2",
    "cdr3",
    "v_call",
    "d_call",
    "j_call",
    "productive",
    "v_identity",
    "d_identity",
    "j_identity",
    "locus",
]

#: Columns that are often redundant or not needed for downstream analysis.
COLS_TO_DROP: list[str] = [
    "light_id",
    "heavy_id",
    "light_id.pasteur",
    "heavy_id.pasteur",
    "light_aid",
    "heavy_aid",
    "light_subset",
    "heavy_subset",
    "light_ighv",
    "heavy_ighv",
]

#: Preferred ordering of high-level metadata fields when present.
META_ORDER: list[str] = ["id", "id.pasteur", "aid", "subset", "ighv_status"]


# ---------------------------------------------------------------------------
# Amino acid set and scales
# ---------------------------------------------------------------------------

#: Canonical set of 20 standard amino acids (one-letter codes).
AA20: set[str] = set("ACDEFGHIKLMNPQRSTVWY")

#: Kyte–Doolittle hydrophobicity scale.
KD: dict[str, float] = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

#: Eisenberg hydrophobicity scale.
EISENBERG: dict[str, float] = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}

#: Boman index contributions per residue (binding potential).
BOMAN: dict[str, float] = {
    "A": 0.17,
    "C": 0.41,
    "D": -1.23,
    "E": -2.02,
    "F": 1.13,
    "G": 0.01,
    "H": -0.96,
    "I": 0.31,
    "K": -0.99,
    "L": 0.56,
    "M": 0.23,
    "N": -0.42,
    "P": -0.31,
    "Q": -0.58,
    "R": -1.01,
    "S": -0.13,
    "T": -0.14,
    "V": 0.07,
    "W": 1.85,
    "Y": 0.94,
}

#: Chou–Fasman helix propensities.
CF_HELIX: dict[str, float] = {
    "A": 1.45,
    "C": 0.77,
    "D": 1.01,
    "E": 1.51,
    "F": 1.13,
    "G": 0.53,
    "H": 1.24,
    "I": 1.08,
    "K": 1.16,
    "L": 1.34,
    "M": 1.20,
    "N": 0.73,
    "P": 0.59,
    "Q": 1.17,
    "R": 0.79,
    "S": 0.79,
    "T": 0.82,
    "V": 1.06,
    "W": 1.14,
    "Y": 0.61,
}

#: Chou–Fasman sheet propensities.
CF_SHEET: dict[str, float] = {
    "A": 0.97,
    "C": 1.30,
    "D": 0.54,
    "E": 0.37,
    "F": 1.38,
    "G": 0.81,
    "H": 0.71,
    "I": 1.60,
    "K": 0.74,
    "L": 1.22,
    "M": 1.67,
    "N": 0.65,
    "P": 0.62,
    "Q": 1.10,
    "R": 0.90,
    "S": 0.72,
    "T": 1.20,
    "V": 1.70,
    "W": 1.19,
    "Y": 1.29,
}

#: TOP-IDP intrinsic disorder scale.
TOP_IDP: dict[str, float] = {
    "A": 0.06,
    "C": -0.22,
    "D": 0.19,
    "E": 0.25,
    "F": -0.15,
    "G": 0.16,
    "H": -0.05,
    "I": -0.20,
    "K": 0.27,
    "L": -0.21,
    "M": -0.09,
    "N": 0.00,
    "P": 0.12,
    "Q": 0.06,
    "R": 0.21,
    "S": 0.10,
    "T": 0.05,
    "V": -0.22,
    "W": -0.23,
    "Y": -0.03,
}


# ---------------------------------------------------------------------------
# pKa values and residue classes
# ---------------------------------------------------------------------------

#: N-terminus pKa (typical peptide).
PKA_N_TERM: float = 9.69

#: C-terminus pKa (typical peptide).
PKA_C_TERM: float = 2.34

#: Side-chain pKa values for ionisable residues.
PKA_SIDE: dict[str, float] = {
    "C": 8.33,
    "D": 3.86,
    "E": 4.25,
    "H": 6.00,
    "K": 10.50,
    "R": 12.50,
    "Y": 10.07,
}

#: Aromatic residues.
AROMATIC: set[str] = set("FYW")

#: Positively charged residues at physiological pH.
POSITIVE: set[str] = set("KRH")

#: Negatively charged residues at physiological pH.
NEGATIVE: set[str] = set("DE")

#: Polar uncharged residues (typical classification).
POLAR_UNCHARGED: set[str] = set("STNQYC")

#: Hydrophobic residues (broad, commonly used set).
HYDROPHOBIC: set[str] = set("AVLIMFWYPGC")

#: Polar residues in a broad sense (uncharged + charged).
POLAR: set[str] = POLAR_UNCHARGED | POSITIVE | NEGATIVE

#: Nonpolar residues (alias for hydrophobic set, kept for backwards compatibility).
NONPOLAR: set[str] = set(HYDROPHOBIC)


# ---------------------------------------------------------------------------
# Hydrogen bond donor and acceptor counts
# ---------------------------------------------------------------------------

#: Approximate count of hydrogen bond donors per residue.
DONORS: dict[str, int] = {
    "K": 1,
    "R": 1,
    "H": 1,
    "W": 1,
    "N": 1,
    "Q": 1,
    "S": 1,
    "T": 1,
    "Y": 1,
    "C": 1,
}

#: Approximate count of hydrogen bond acceptors per residue.
ACCEPTORS: dict[str, int] = {
    "D": 1,
    "E": 1,
    "N": 1,
    "Q": 1,
    "H": 1,
    "S": 1,
    "T": 1,
    "Y": 1,
}

