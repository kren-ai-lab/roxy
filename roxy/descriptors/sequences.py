
"""High-level utilities for computing global protein sequence descriptors.

This module provides the :class:`ProteinDescriptorError` class, which exposes
convenience methods to featurise individual protein sequences or collections
of sequences (pandas DataFrames and FASTA files).

The implementation is intentionally lightweight and dependency-minimal so that
it can be reused as a backend in different projects (e.g. the Roxy feature
engineering library).

The descriptors are primarily global, sequence-level summaries such as:

- sequence length and amino-acid composition
- GRAVY (Kyte–Doolittle) and Eisenberg hydrophobicity averages
- aromatic, charged, polar and hydrophobic residue fractions
- mean TOP-IDP disorder score
- Chou–Fasman helix and sheet propensities (mean values)
- Boman index (binding potential)
- net charge at a given pH, FCR and NCPR
- approximate hydrogen-bond donor and acceptor counts (per residue)
- Shannon entropy of amino-acid usage
- simple linguistic complexity measures (k-mer based, k = 1..3)
- optional AAIndex-based mean properties for user-specified indices

All methods assume input sequences are strings of one-letter amino-acid codes.
Unknown or non-standard characters are ignored in scale-based calculations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import math
import pandas as pd

try:
    # Preferred location when used inside the Roxy ecosystem
    from roxy.core.constants import (
        AA20,
        KD,
        EISENBERG,
        AROMATIC,
        POLAR,
        NONPOLAR,
        POSITIVE,
        NEGATIVE,
        TOP_IDP,
        CF_HELIX,
        CF_SHEET,
        PKA_N_TERM,
        PKA_C_TERM,
        PKA_SIDE,
        BOMAN,
        DONORS,
        ACCEPTORS,
    )
except ImportError:  # pragma: no cover - fallback for standalone usage
    from core.constants import (  # type: ignore[no-redef]
        AA20,
        KD,
        EISENBERG,
        AROMATIC,
        POLAR,
        NONPOLAR,
        POSITIVE,
        NEGATIVE,
        TOP_IDP,
        CF_HELIX,
        CF_SHEET,
        PKA_N_TERM,
        PKA_C_TERM,
        PKA_SIDE,
        BOMAN,
        DONORS,
        ACCEPTORS,
    )

# AAIndex integration (optional fallback if core.aaindex is not available)
try:
    from roxy.core.aaindex import compute_aaindex_means_for_sequence
except ImportError:  # pragma: no cover
    try:
        from core.aaindex import compute_aaindex_means_for_sequence  # type: ignore[assignment]
    except ImportError:  # pragma: no cover
        compute_aaindex_means_for_sequence = None  # type: ignore[assignment]

from dataclasses import dataclass
from typing import Any

from .base import BaseDescriptorEngine

class ProteinDescriptorError:
    """Compute global protein sequence descriptors.

    This class is organised using class methods so that it can be used either
    directly (e.g. :meth:`featurize_sequence`) or wrapped inside higher-level
    descriptor engines. No instance state is required.

    Notes
    -----
    - All methods treat the input sequence as a simple string of one-letter
      amino-acid codes. Lowercase letters are normalised to uppercase.
    - Non-standard characters (e.g. X, B, Z, gaps) are ignored in calculations
      that rely on amino-acid-specific scales. Length-based quantities still
      use the full sequence length.
    - Charge calculations are based on a simple Henderson–Hasselbalch model
      with pKa values defined in :mod:`roxy.core.constants` (or `core.constants`
      when used standalone).
    - Optional AAIndex-based descriptors can be enabled by passing
      ``aaindex_codes`` to :meth:`featurize_sequence`, in which case the
      AAIndex CSV must be available and loadable via ``roxy.core.aaindex``.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _clean(cls, seq: str) -> str:
        """Normalise a sequence string.

        The cleaning step performs:
        - conversion to string
        - stripping of leading/trailing whitespace
        - conversion to uppercase
        - removal of terminal stop codons ("*")

        Parameters
        ----------
        seq:
            Raw sequence input.

        Returns
        -------
        str
            Cleaned sequence.
        """
        s = (seq or "").strip().upper().replace("*", "")
        return s

    # --------------------------- composition ---------------------------

    @classmethod
    def _aa_counts(cls, seq: str) -> Dict[str, int]:
        """Count occurrences of each canonical amino acid in the sequence."""
        s = cls._clean(seq)
        return {aa: s.count(aa) for aa in AA20}

    @classmethod
    def _aa_fractions(cls, seq: str) -> Dict[str, float]:
        """Compute amino-acid fractions for the 20 canonical residues."""
        s = cls._clean(seq)
        L = len(s)
        if L == 0:
            return {f"aa_frac_{aa}": 0.0 for aa in AA20}
        return {f"aa_frac_{aa}": s.count(aa) / L for aa in AA20}

    # --------------------------- scales & indices ---------------------------

    @classmethod
    def _mean_scale(cls, seq: str, scale: Dict[str, float]) -> float:
        """Compute the mean value of a per-residue scale for a sequence.

        Residues not present in the scale are ignored. If no residue in the
        sequence is present in the scale, NaN is returned.

        Parameters
        ----------
        seq:
            Amino-acid sequence.
        scale:
            Mapping from residue code to scale value.

        Returns
        -------
        float
            Mean scale value, or NaN if not computable.
        """
        s = cls._clean(seq)
        values = [scale[aa] for aa in s if aa in scale]
        if not values:
            return math.nan
        return float(sum(values) / len(values))

    @classmethod
    def _shannon_entropy(cls, seq: str) -> float:
        """Compute Shannon entropy of amino-acid usage.

        The entropy is computed in bits (log base 2) over the 20 canonical
        amino acids present in the sequence. If the sequence is empty,
        NaN is returned.

        Parameters
        ----------
        seq:
            Amino-acid sequence.

        Returns
        -------
        float
            Shannon entropy in bits.
        """
        s = cls._clean(seq)
        L = len(s)
        if L == 0:
            return math.nan
        # frequency of each residue present in the sequence
        probs = []
        for aa in AA20:
            count = s.count(aa)
            if count:
                p = count / L
                probs.append(p)
        if not probs:
            return math.nan
        return float(-sum(p * math.log2(p) for p in probs))

    @classmethod
    def _linguistic_complexity(cls, seq: str, max_k: int = 3) -> Dict[str, float]:
        """Compute simple k-mer based linguistic complexity metrics.

        For each k in ``1..max_k``, the complexity is defined as:

        .. math::

            C_k = \\frac{N_k^{\\text{obs}}}{\\min(20^k, L - k + 1)}

        where :math:`N_k^{obs}` is the number of distinct k-mers observed in
        the sequence, and :math:`L` is the sequence length.

        If ``L < k``, the complexity for that k is defined as 0.0.

        Parameters
        ----------
        seq:
            Input amino-acid sequence.
        max_k:
            Maximum k-mer length to consider.

        Returns
        -------
        dict
            Mapping from feature name (e.g. ``lc_k1``) to complexity value.
        """
        s = cls._clean(seq)
        L = len(s)
        if L == 0:
            return {f"lc_k{k}": 0.0 for k in range(1, max_k + 1)}

        feats: Dict[str, float] = {}
        for k in range(1, max_k + 1):
            if L < k:
                feats[f"lc_k{k}"] = 0.0
                continue
            observed = {s[i : i + k] for i in range(L - k + 1)}
            max_possible = min(20 ** k, L - k + 1)
            feats[f"lc_k{k}"] = len(observed) / max_possible if max_possible > 0 else 0.0
        return feats

    # --------------------------- charge & related ---------------------------

    @classmethod
    def _net_charge(cls, seq: str, pH: float, include_histidine: bool = False) -> float:
        """Estimate the net charge of a sequence at a given pH.

        The calculation uses a simple Henderson–Hasselbalch approach with
        pKa values for the N-terminus, C-terminus and ionisable side chains
        defined in :data:`PKA_N_TERM`, :data:`PKA_C_TERM` and :data:`PKA_SIDE`.

        Parameters
        ----------
        seq:
            Input amino-acid sequence.
        pH:
            Solution pH at which to estimate the net charge.
        include_histidine:
            If True, histidine is treated as a positively charged residue.
            If False, histidine is excluded from the net-charge calculation
            (which may be desirable for certain coarse descriptors).

        Returns
        -------
        float
            Estimated net charge at the requested pH.
        """
        s = cls._clean(seq)
        if not s:
            return 0.0

        nK = s.count("K")
        nR = s.count("R")
        nH = s.count("H")
        nD = s.count("D")
        nE = s.count("E")
        nC = s.count("C")
        nY = s.count("Y")

        # N- and C-terminus
        nterm = 1.0 / (1.0 + 10.0 ** (pH - PKA_N_TERM))
        cterm = 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - pH))

        # Side chains (fractional charges)
        K = nK * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["K"])))
        R = nR * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["R"])))
        H = nH * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["H"]))) if include_histidine else 0.0
        D = nD * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["D"] - pH)))
        E = nE * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["E"] - pH)))
        C = nC * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["C"] - pH)))
        Y = nY * (1.0 / (1.0 + 10.0 ** (PKA_SIDE["Y"] - pH)))

        positive = nterm + K + R + H
        negative = cterm + D + E + C + Y

        return float(positive - negative)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def featurize_sequence(
        cls,
        seq: str,
        *,
        pH: float = 7.0,
        include_histidine_in_charge: bool = False,
        aaindex_codes: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        """Compute global descriptors for a single protein sequence.

        Parameters
        ----------
        seq:
            Amino-acid sequence (one-letter codes). Lowercase letters are
            accepted and internally normalised to uppercase.
        pH:
            pH at which to estimate charge-related descriptors.
        include_histidine_in_charge:
            Whether to include histidine as a positively charged residue in
            the net-charge calculation.
        aaindex_codes:
            Optional iterable of AAIndex identifiers (i.e. values of the
            ``index_id`` column in the AAIndex CSV). If provided, for each
            code a descriptor named ``aaindex_<CODE>_mean`` will be added
            to the feature dictionary, representing the mean AAIndex value
            over the sequence.

        Returns
        -------
        dict
            Mapping from feature name to value.
        """
        s = cls._clean(seq)
        L = len(s)

        feats: Dict[str, float] = {}

        # Basic length and composition
        feats["length"] = float(L)
        feats.update(cls._aa_fractions(s))

        if L == 0:
            # For empty sequences, all remaining descriptors are NaN or 0
            feats.update(
                {
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
            )
            # AAIndex still returns NaN features if requested
            if aaindex_codes is not None and compute_aaindex_means_for_sequence is not None:
                feats.update(
                    {
                        f"aaindex_{code}_mean": math.nan
                        for code in aaindex_codes
                    }
                )
            return feats

        # Residue class fractions
        n_aromatic = sum(s.count(a) for a in AROMATIC)
        n_pos = sum(s.count(a) for a in POSITIVE)
        n_neg = sum(s.count(a) for a in NEGATIVE)
        n_polar = sum(s.count(a) for a in POLAR)
        n_nonpolar = sum(s.count(a) for a in NONPOLAR)

        feats["frac_aromatic"] = n_aromatic / L
        feats["frac_positive"] = n_pos / L
        feats["frac_negative"] = n_neg / L
        feats["frac_polar"] = n_polar / L
        feats["frac_nonpolar"] = n_nonpolar / L

        # Scale-based averages
        feats["gravy_kd"] = cls._mean_scale(s, KD)
        feats["hydropathy_eisenberg"] = cls._mean_scale(s, EISENBERG)
        feats["top_idp_mean"] = cls._mean_scale(s, TOP_IDP)
        feats["helix_propensity_mean"] = cls._mean_scale(s, CF_HELIX)
        feats["sheet_propensity_mean"] = cls._mean_scale(s, CF_SHEET)

        # Boman index (sum of per-residue contributions / length)
        boman_sum = sum(BOMAN.get(aa, 0.0) for aa in s)
        feats["boman_index"] = boman_sum / L

        # Charge-related descriptors
        net_charge = cls._net_charge(s, pH=pH, include_histidine=include_histidine_in_charge)
        feats["net_charge_pH"] = net_charge
        feats["fcr"] = (n_pos + n_neg) / L
        feats["ncpr"] = net_charge / L if L > 0 else 0.0

        # Donors / acceptors (per residue)
        n_donors = sum(DONORS.get(aa, 0) for aa in s)
        n_acceptors = sum(ACCEPTORS.get(aa, 0) for aa in s)
        feats["donors_per_residue"] = n_donors / L
        feats["acceptors_per_residue"] = n_acceptors / L

        # Entropy and linguistic complexity
        feats["aa_entropy"] = cls._shannon_entropy(s)
        feats.update(cls._linguistic_complexity(s, max_k=3))

        # AAIndex-based descriptors
        if aaindex_codes is not None:
            if compute_aaindex_means_for_sequence is None:
                raise RuntimeError(
                    "AAIndex support is not available. Make sure `roxy.core.aaindex` "
                    "or `core.aaindex` is importable and AAIndex is configured."
                )
            aa_feats = compute_aaindex_means_for_sequence(seq, aaindex_codes)
            feats.update(aa_feats)

        return feats

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @classmethod
    def featurize_df(
        cls,
        df: pd.DataFrame,
        *,
        sequence_col: str = "sequence",
        id_col: Optional[str] = None,
        pH: float = 7.0,
        include_histidine_in_charge: bool = False,
        aaindex_codes: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Featurise all sequences in a DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame containing at least a column with amino-acid
            sequences.
        sequence_col:
            Name of the column containing sequences.
        id_col:
            Optional name of a column to be carried over as an identifier.
            If provided, this column will be included in the output and
            used as the index; otherwise the original DataFrame index is
            preserved.
        pH:
            pH at which to estimate charge-related descriptors.
        include_histidine_in_charge:
            Whether to include histidine in the net-charge calculation.
        aaindex_codes:
            Optional iterable of AAIndex identifiers to be passed to
            :meth:`featurize_sequence`.

        Returns
        -------
        pandas.DataFrame
            DataFrame where each row corresponds to one input sequence
            and each column corresponds to a descriptor. If `id_col` is
            provided, it is included in the output.
        """
        if sequence_col not in df.columns:
            raise ValueError(f"Column {sequence_col!r} not found in input DataFrame.")

        rows: List[Dict[str, float]] = []
        index: List[object] = []

        for idx, row in df.iterrows():
            seq = row[sequence_col]
            feats = cls.featurize_sequence(
                seq,
                pH=pH,
                include_histidine_in_charge=include_histidine_in_charge,
                aaindex_codes=aaindex_codes,
            )
            if id_col is not None:
                if id_col not in df.columns:
                    raise ValueError(f"Column {id_col!r} not found in input DataFrame.")
                feats[id_col] = row[id_col]
                index.append(row[id_col])
            else:
                index.append(idx)
            rows.append(feats)

        feat_df = pd.DataFrame(rows, index=index)

        # If we included an id_col as a feature, keep it as a column but not as index name
        if id_col is not None and id_col in feat_df.columns:
            feat_df.index.name = id_col

        return feat_df

    @classmethod
    def featurize_fasta(
        cls,
        fasta_path: str,
        *,
        pH: float = 7.0,
        include_histidine_in_charge: bool = False,
        aaindex_codes: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Featurise all sequences contained in a FASTA file.

        Parameters
        ----------
        fasta_path:
            Path to a FASTA file containing one or more protein sequences.
            Header lines (starting with ">") are interpreted as sequence
            identifiers.
        pH:
            pH at which to estimate charge-related descriptors.
        include_histidine_in_charge:
            Whether to include histidine in the net-charge calculation.
        aaindex_codes:
            Optional iterable of AAIndex identifiers to be passed to
            :meth:`featurize_sequence`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per FASTA entry. The output contains
            an ``id`` column with the FASTA headers (without the leading
            ">"), a ``sequence`` column with the raw sequences, and all
            computed descriptor columns.
        """
        ids: List[str] = []
        seqs: List[str] = []

        current_id: Optional[str] = None
        current_seq: List[str] = []

        with open(fasta_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    # Flush previous sequence
                    if current_id is not None:
                        ids.append(current_id)
                        seqs.append("".join(current_seq))
                        current_seq = []
                    current_id = line[1:].strip()
                else:
                    current_seq.append(line.strip())

        # Flush last entry
        if current_id is not None:
            ids.append(current_id)
            seqs.append("".join(current_seq))

        rows: List[Dict[str, float]] = []
        for seq_id, seq in zip(ids, seqs):
            feats = cls.featurize_sequence(
                seq,
                pH=pH,
                include_histidine_in_charge=include_histidine_in_charge,
                aaindex_codes=aaindex_codes,
            )
            feats["id"] = seq_id
            feats["sequence"] = cls._clean(seq)
            rows.append(feats)

        df = pd.DataFrame(rows)
        # Order columns: id, sequence, then descriptors
        col_order = ["id", "sequence"] + [c for c in df.columns if c not in ("id", "sequence")]
        return df[col_order]

@dataclass
class GlobalSequenceDescriptors(BaseDescriptorEngine):
    """Roxy descriptor engine for global sequence features (with optional AAIndex).

    This class is the bridge between the low-level feature computation
    implemented in :class:`ProteinDescriptorError` and the generic
    descriptor engine interface used by Roxy.

    It can be registered under a short name (e.g. ``"seq_global"``) and
    combined with other engines through :mod:`roxy.descriptors.registry`.

    Parameters
    ----------
    name :
        Engine name used for registration and logging.
    pH :
        pH at which to estimate charge-related descriptors.
    include_histidine_in_charge :
        Whether to include histidine in the net-charge calculation.
    aaindex_codes :
        Optional iterable of AAIndex identifiers. If provided, additional
        features with names ``aaindex_<CODE>_mean`` will be computed for
        each sequence.
    """

    name: str = "global_sequence_descriptors"
    pH: float = 7.0
    include_histidine_in_charge: bool = False
    aaindex_codes: Optional[Iterable[str]] = None

    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptors for all sequences in the samples table.

        Parameters
        ----------
        samples :
            DataFrame with at least a ``sequence`` column containing
            amino-acid sequences (one-letter codes).

        Returns
        -------
        pandas.DataFrame
            Feature table with one row per sample and one column per
            descriptor. The index is aligned to ``samples.index``.
        """
        if "sequence" not in samples.columns:
            raise ValueError("Expected a 'sequence' column in samples.")

        rows: List[Dict[str, Any]] = []

        for idx, row in samples.iterrows():
            seq = row["sequence"]
            feats = ProteinDescriptorError.featurize_sequence(
                seq,
                pH=self.pH,
                include_histidine_in_charge=self.include_histidine_in_charge,
                aaindex_codes=self.aaindex_codes,
            )
            feats["_index"] = idx
            rows.append(feats)

        df = pd.DataFrame(rows).set_index("_index")
        return df
