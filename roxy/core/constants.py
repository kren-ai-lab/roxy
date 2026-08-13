"""Constants and lookup tables for protein sequence descriptor calculations.

This module centralises amino-acid level constants that are reused across
the protein descriptor toolbox. It intentionally contains **no business
logic**, only data, so that it can be safely imported from different parts of
the codebase (e.g. descriptor engines, analysis utilities, etc.).

The constants defined here include:

- Column name lists used when working with antibody or chain-level datasets.
- The canonical set of 20 amino acids (`AA20`).
- Hydrophobicity scales (Kyte-Doolittle, Eisenberg normalized consensus).
- Boman index contributions.
- Chou-Fasman helix and sheet propensities.
- TOP-IDP intrinsic disorder scale.
- pKa values for termini and side chains.
- Residue class sets (aromatic, charged, polar, hydrophobic).
- Hydrogen-bond donor and acceptor residue sets.

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

#: Canonical ordered tuple of 20 standard amino acids (alphabetical).
AA20_ORDERED: tuple[str, ...] = tuple(sorted(AA20))

#: Kyte-Doolittle hydrophobicity scale.
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

#: Eisenberg normalized consensus hydrophobicity scale (Eisenberg et al., 1984).
#: Note: the AAindex entry EISD840101 is titled "Consensus normalized
#: hydrophobicity scale" but its values are *not* the normalized ones; the
#: table below is the normalized consensus scale.
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

#: Hopp-Woods hydrophilicity scale.
HYDROPHILICITY: dict[str, float] = {
    "A": -0.50,
    "C": -1.00,
    "D": 3.00,
    "E": 3.00,
    "F": -2.50,
    "G": 0.00,
    "H": -0.50,
    "I": -1.80,
    "K": 3.00,
    "L": -1.80,
    "M": -1.30,
    "N": 0.20,
    "P": 0.00,
    "Q": 0.20,
    "R": 3.00,
    "S": 0.30,
    "T": -0.40,
    "V": -1.50,
    "W": -3.40,
    "Y": -2.30,
}

#: Side-chain mass scale (Da).
SIDECHAIN_MASS: dict[str, float] = {
    "A": 15.0,
    "C": 47.0,
    "D": 59.0,
    "E": 73.0,
    "F": 91.0,
    "G": 1.0,
    "H": 82.0,
    "I": 57.0,
    "K": 72.0,
    "L": 57.0,
    "M": 75.0,
    "N": 58.0,
    "P": 41.0,
    "Q": 72.0,
    "R": 100.0,
    "S": 31.0,
    "T": 45.0,
    "V": 43.0,
    "W": 130.0,
    "Y": 107.0,
}

#: Average volumes of residues (Å³), Pontius et al. 1996 — AAindex PONJ960101.
VOLUME: dict[str, float] = {
    "A": 91.5,
    "C": 114.4,
    "D": 135.2,
    "E": 154.6,
    "F": 198.8,
    "G": 67.5,
    "H": 163.2,
    "I": 162.6,
    "K": 162.5,
    "L": 163.4,
    "M": 165.9,
    "N": 138.3,
    "P": 123.4,
    "Q": 156.4,
    "R": 196.1,
    "S": 102.0,
    "T": 126.0,
    "V": 138.4,
    "W": 209.8,
    "Y": 237.2,
}

#: Residue average molecular weights (Da).
AA_MOLECULAR_WEIGHT: dict[str, float] = {
    "A": 89.09,
    "C": 121.15,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

#: Residue solubility values used by the Boman (protein interaction) index,
#: Boman 2003 — same table as ``Peptides::boman`` in R. The index itself is
#: ``-sum(values) / len(seq)``, so the sign is flipped at the call site.
#: Proline is absent from the published scale; it is kept here as 0.0 so that
#: it still counts towards the sequence length, matching the R implementation
#: (``na.rm = TRUE`` over ``length(seq)``).
BOMAN: dict[str, float] = {
    "A": 1.81,
    "C": 1.28,
    "D": -8.72,
    "E": -6.81,
    "F": 2.98,
    "G": 0.94,
    "H": -4.66,
    "I": 4.92,
    "K": -5.55,
    "L": 4.92,
    "M": 2.35,
    "N": -6.64,
    "P": 0.00,
    "Q": -5.54,
    "R": -14.92,
    "S": -3.40,
    "T": -2.57,
    "V": 4.04,
    "W": 2.33,
    "Y": -0.14,
}

#: Polarity (Zimmerman et al., 1968) — AAindex ZIMJ680103.
POLARITY: dict[str, float] = {
    "A": 0.00,
    "C": 1.48,
    "D": 49.70,
    "E": 49.90,
    "F": 0.35,
    "G": 0.00,
    "H": 51.60,
    "I": 0.13,
    "K": 49.50,
    "L": 0.13,
    "M": 1.43,
    "N": 3.38,
    "P": 1.58,
    "Q": 3.53,
    "R": 52.00,
    "S": 1.67,
    "T": 1.66,
    "V": 0.13,
    "W": 2.10,
    "Y": 1.61,
}

#: Bhaskaran-Ponnuswamy backbone flexibility scale.
FLEXIBILITY: dict[str, float] = {
    "A": 0.357,
    "C": 0.346,
    "D": 0.511,
    "E": 0.497,
    "F": 0.314,
    "G": 0.544,
    "H": 0.323,
    "I": 0.462,
    "K": 0.466,
    "L": 0.365,
    "M": 0.295,
    "N": 0.463,
    "P": 0.509,
    "Q": 0.493,
    "R": 0.529,
    "S": 0.507,
    "T": 0.444,
    "V": 0.386,
    "W": 0.305,
    "Y": 0.420,
}

#: Normalized frequency of alpha-helix (Chou-Fasman, 1978b) — AAindex CHOP780201.
CF_HELIX: dict[str, float] = {
    "A": 1.42,
    "C": 0.70,
    "D": 1.01,
    "E": 1.51,
    "F": 1.13,
    "G": 0.57,
    "H": 1.00,
    "I": 1.08,
    "K": 1.16,
    "L": 1.21,
    "M": 1.45,
    "N": 0.67,
    "P": 0.57,
    "Q": 1.11,
    "R": 0.98,
    "S": 0.77,
    "T": 0.83,
    "V": 1.06,
    "W": 1.08,
    "Y": 0.69,
}

#: Normalized frequency of beta-sheet (Chou-Fasman, 1978b) — AAindex CHOP780202.
CF_SHEET: dict[str, float] = {
    "A": 0.83,
    "C": 1.19,
    "D": 0.54,
    "E": 0.37,
    "F": 1.38,
    "G": 0.75,
    "H": 0.87,
    "I": 1.60,
    "K": 0.74,
    "L": 1.30,
    "M": 1.05,
    "N": 0.89,
    "P": 0.55,
    "Q": 1.10,
    "R": 0.93,
    "S": 0.75,
    "T": 1.19,
    "V": 1.70,
    "W": 1.37,
    "Y": 1.47,
}

#: Normalized frequency of beta-turn (Chou-Fasman, 1978b) — AAindex CHOP780203.
CF_TURN: dict[str, float] = {
    "A": 0.74,
    "C": 0.96,
    "D": 1.52,
    "E": 0.95,
    "F": 0.66,
    "G": 1.56,
    "H": 0.95,
    "I": 0.47,
    "K": 1.19,
    "L": 0.50,
    "M": 0.60,
    "N": 1.46,
    "P": 1.56,
    "Q": 0.96,
    "R": 1.01,
    "S": 1.43,
    "T": 0.98,
    "V": 0.59,
    "W": 0.60,
    "Y": 1.14,
}

#: TOP-IDP intrinsic disorder scale (Campen et al., 2008 — Table 2).
TOP_IDP: dict[str, float] = {
    "A": 0.06,
    "C": 0.02,
    "D": 0.192,
    "E": 0.736,
    "F": -0.697,
    "G": 0.166,
    "H": 0.303,
    "I": -0.486,
    "K": 0.586,
    "L": -0.326,
    "M": -0.397,
    "N": 0.007,
    "P": 0.987,
    "Q": 0.318,
    "R": 0.180,
    "S": 0.341,
    "T": 0.059,
    "V": -0.121,
    "W": -0.884,
    "Y": -0.510,
}


# ---------------------------------------------------------------------------
# pKa values and residue classes
# ---------------------------------------------------------------------------

#: N-terminus pKa: mean of the alpha-amino pKa of the 20 standard amino acids
#: (Lehninger, table 3-1).
PKA_N_TERM: float = 9.47

#: C-terminus pKa: mean of the alpha-carboxyl pKa of the 20 standard amino acids
#: (Lehninger, table 3-1).
PKA_C_TERM: float = 2.1555

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

#: Canonical amino-acid group assignments (15 groups) used by composition descriptors.
AA_GROUPS: dict[str, set[str]] = {
    "positive": set("KRH"),
    "negative": set("DE"),
    "charged": set("KRHDE"),
    "polar": set("STNQCYWHKRDE"),
    "nonpolar": set("AVLIMFGP"),
    "aromatic": set("FWYH"),
    "tiny": set("AGCS"),
    "small": set("AGCSTVPDN"),
    "sulfur": set("CM"),
    "hydroxyl": set("STY"),
    "amide": set("NQ"),
    "hydrophobic": set("ILVMFYWHKCAT"),
    "hydrophilic": set("RNDQES"),
    "disorder_promoting": set("SAPREKG"),
    "order_promoting": set("CWYFILV"),
}


# ---------------------------------------------------------------------------
# Hydrogen bond donor and acceptor residues
# ---------------------------------------------------------------------------

#: Residues whose side chain can donate a hydrogen bond (AAindex FAUJ880109).
DONORS: frozenset[str] = frozenset("KRHWNQSTYDE")

#: Residues whose side chain can accept a hydrogen bond.
ACCEPTORS: frozenset[str] = frozenset("DENQHSTY")
