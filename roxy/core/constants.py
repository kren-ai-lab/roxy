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

from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Column name conventions for antibody / chain datasets
# ---------------------------------------------------------------------------

#: Metadata columns typically associated with antibody-level identifiers.
META_COLS: List[str] = ["id", "id.pasteur", "aid", "subset", "ighv"]

#: Columns commonly found in chain-level (e.g. heavy/light chain) tables.
CHAIN_COLS: List[str] = [
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
COLS_TO_DROP: List[str] = [
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
META_ORDER: List[str] = ["id", "id.pasteur", "aid", "subset", "ighv_status"]


# ---------------------------------------------------------------------------
# Amino acid set and scales
# ---------------------------------------------------------------------------

#: Canonical ordered tuple of the 20 standard amino acids.
STANDARD_AMINO_ACID_ORDER: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

#: Stable canonical amino-acid order shared by descriptor families.
CANONICAL_AMINO_ACID_ORDER: tuple[str, ...] = STANDARD_AMINO_ACID_ORDER

#: Canonical set of 20 standard amino acids (one-letter codes).
AA20: Set[str] = set(STANDARD_AMINO_ACID_ORDER)

#: Common ambiguous or non-canonical amino-acid symbols seen in protein data.
AMBIGUOUS_AMINO_ACIDS: Set[str] = set("BJOUXZ")

#: Conventional stop marker used in protein sequence strings.
SEQUENCE_STOP_MARKER: str = "*"

#: Kyte–Doolittle hydrophobicity scale.
KD: Dict[str, float] = {
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
EISENBERG: Dict[str, float] = {
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

#: Grantham polarity scale.
POLARITY: Dict[str, float] = {
    "A": 8.1,
    "R": 10.5,
    "N": 11.6,
    "D": 13.0,
    "C": 5.5,
    "Q": 10.5,
    "E": 12.3,
    "G": 9.0,
    "H": 10.4,
    "I": 5.2,
    "L": 4.9,
    "K": 11.3,
    "M": 5.7,
    "F": 5.2,
    "P": 8.0,
    "S": 9.2,
    "T": 8.6,
    "W": 5.4,
    "Y": 6.2,
    "V": 5.9,
}

#: Approximate residue masses in Daltons for peptide residues.
RESIDUE_MASS: Dict[str, float] = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "Q": 128.1307,
    "E": 129.1155,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

#: Boman index contributions per residue (binding potential).
BOMAN: Dict[str, float] = {
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
CF_HELIX: Dict[str, float] = {
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
CF_SHEET: Dict[str, float] = {
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
TOP_IDP: Dict[str, float] = {
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
PKA_SIDE: Dict[str, float] = {
    "C": 8.33,
    "D": 3.86,
    "E": 4.25,
    "H": 6.00,
    "K": 10.50,
    "R": 12.50,
    "Y": 10.07,
}

#: Aromatic residues.
AROMATIC: Set[str] = set("FYW")

#: Aliphatic residues.
ALIPHATIC: Set[str] = set("AILV")

#: Positively charged residues at physiological pH.
POSITIVE: Set[str] = set("KRH")

#: Negatively charged residues at physiological pH.
NEGATIVE: Set[str] = set("DE")

#: Polar uncharged residues (typical classification).
POLAR_UNCHARGED: Set[str] = set("STNQYC")

#: Hydrophobic residues (broad, commonly used set).
HYDROPHOBIC: Set[str] = set("AVLIMFWYPGC")

#: Polar residues in a broad sense (uncharged + charged).
POLAR: Set[str] = POLAR_UNCHARGED | POSITIVE | NEGATIVE

#: Nonpolar residues (alias for hydrophobic set, kept for backwards compatibility).
NONPOLAR: Set[str] = set(HYDROPHOBIC)

#: Charged residues at physiological pH.
CHARGED: Set[str] = set(POSITIVE | NEGATIVE)

#: Very small side-chain residues.
TINY: Set[str] = set("ACGST")

#: Small residues with compact side chains.
SMALL: Set[str] = set("ACDGNPSTV")

#: Branched aliphatic residues.
BRANCHED: Set[str] = set("ILV")

#: Sulfur-containing residues.
SULFUR: Set[str] = set("CM")

#: Hydroxyl-bearing residues.
HYDROXYL: Set[str] = set("STY")

#: Amide side-chain residues.
AMIDE: Set[str] = set("NQ")

#: Broad hydrophilic residue set distinct from the wider polar group.
HYDROPHILIC: Set[str] = set("RNDQEHKST")

#: Disorder-promoting residues based on positive TOP-IDP values.
DISORDER_PROMOTING: Set[str] = set("ADEGKPQRST")

#: Order-promoting residues based on negative TOP-IDP values.
ORDER_PROMOTING: Set[str] = set("CFHILMVWY")

#: Default grouped-composition families in stable output order.
DEFAULT_GROUPED_RESIDUE_SETS: Dict[str, frozenset[str]] = {
    "positive": frozenset(POSITIVE),
    "negative": frozenset(NEGATIVE),
    "charged": frozenset(CHARGED),
    "polar": frozenset(POLAR),
    "nonpolar": frozenset(NONPOLAR),
    "aromatic": frozenset(AROMATIC),
    "aliphatic": frozenset(ALIPHATIC),
    "tiny": frozenset(TINY),
    "small": frozenset(SMALL),
    "branched": frozenset(BRANCHED),
    "sulfur": frozenset(SULFUR),
    "hydroxyl": frozenset(HYDROXYL),
    "amide": frozenset(AMIDE),
    "hydrophobic": frozenset(HYDROPHOBIC),
    "hydrophilic": frozenset(HYDROPHILIC),
    "disorder_promoting": frozenset(DISORDER_PROMOTING),
    "order_promoting": frozenset(ORDER_PROMOTING),
}

#: Stable default order for grouped-composition residue families.
DEFAULT_GROUPED_RESIDUE_ORDER: tuple[str, ...] = tuple(DEFAULT_GROUPED_RESIDUE_SETS)


# ---------------------------------------------------------------------------
# Hydrogen bond donor and acceptor counts
# ---------------------------------------------------------------------------

#: Approximate count of hydrogen bond donors per residue.
DONORS: Dict[str, int] = {
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
ACCEPTORS: Dict[str, int] = {
    "D": 1,
    "E": 1,
    "N": 1,
    "Q": 1,
    "H": 1,
    "S": 1,
    "T": 1,
    "Y": 1,
}

# ---------------------------------------------------------------------------
# AAIndex configuration
# ---------------------------------------------------------------------------

#: Default subdirectory name under the user cache directory for Roxy.
ROXY_CACHE_SUBDIR: str = "roxy"

#: Default filename for the cached AAIndex CSV.
AAINDEX_FILENAME: str = "aaindex.csv"

#: Default URL to download the AAIndex CSV file.
#: This points directly to the CSV hosted on Google Drive.
AAINDEX_URL: str = (
    "https://drive.google.com/uc"
    "?export=download&id=1On3-2vQh7BBy5VHk87nlB746rWiMrwqp"
)
