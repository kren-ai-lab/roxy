
"""Global protein sequence descriptors for Roxy.

This module provides utilities to compute *global* descriptors for
protein sequences and exposes them as a Roxy descriptor engine:

- :class:`ProteinSequenceDescriptors` –
  low-level static/class methods to featurise individual sequences,
  DataFrames or FASTA files.
- :class:`GlobalSequenceDescriptors` –
  Roxy-compatible engine that plugs into :class:`roxy.core.dataset.RoxyDataset`
  and the descriptor registry.

The descriptors incluyen, entre otros:

- length and amino-acid composition
- GRAVY (Kyte–Doolittle) and Eisenberg hydrophobicity
- aromatic / charged / polar / nonpolar residue fractions
- TOP-IDP disorder, Chou–Fasman helix / sheet propensities
- Boman index
- net charge at a given pH, FCR, NCPR
- donors/acceptors per residue
- Shannon entropy of AA usage
- simple k-mer linguistic complexity (k = 1..3)
- optional AAIndex-based mean properties for specified indices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import math
import pandas as pd

# Constants (prefer Roxy location, fall back to standalone usage)
try:
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

from roxy.core.exceptions import DescriptorError, AAIndexError
from roxy.core.logging_utils import get_logger

from .base import BaseDescriptorEngine

logger = get_logger(__name__)


class ProteinSequenceDescriptors:
    """Compute global protein sequence descriptors.

    This class is organised using class methods so that it can be used either
    directly (e.g. :meth:`featurize_sequence`) or wrapped inside higher-level
    descriptor engines. No instance state is required.

    Notes
    -----
    - Input sequences are assumed to be strings of one-letter amino-acid
      codes. Lowercase letters are accepted and normalised to uppercase.
    - Non-standard characters (e.g. X, B, Z, gaps) are ignored in
      scale-based calculations. Length still uses the full sequence.
    - Net charge is estimated with a simple Henderson–Hasselbalch model
      using pKa values defined in :mod:`roxy.core.constants`.
    - Optional AAIndex-based descriptors can be enabled by passing
      ``aaindex_codes`` to :meth:`featurize_sequence`. In that case, the
      AAIndex CSV must be available via :mod:`roxy.core.aaindex`.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _clean(cls, seq: str) -> str:
        """Normalise a sequence string."""
        s = (seq or "").strip().upper().replace("*", "")
        return s

    @classmethod
    def _aa_fractions(cls, seq: str) -> Dict[str, float]:
        """Compute amino-acid fractions for the 20 canonical residues."""
        s = cls._clean(seq)
        L = len(s)
        if L == 0:
            return {f"aa_frac_{aa}": 0.0 for aa in AA20}
        return {f"aa_frac_{aa}": s.count(aa) / L for aa in AA20}

    @classmethod
    def _mean_scale(cls, seq: str, scale: Dict[str, float]) -> float:
        """Mean value of a residue-level scale over a sequence."""
        s = cls._clean(seq)
        values = [scale[aa] for aa in s if aa in scale]
        if not values:
            return math.nan
        return float(sum(values) / len(values))

    @classmethod
    def _shannon_entropy(cls, seq: str) -> float:
        """Shannon entropy (bits) of amino-acid usage."""
        s = cls._clean(seq)
        L = len(s)
        if L == 0:
            return math.nan
        probs: List[float] = []
        for aa in AA20:
            count = s.count(aa)
            if count:
                probs.append(count / L)
        if not probs:
            return math.nan
        return float(-sum(p * math.log2(p) for p in probs))

    @classmethod
    def _linguistic_complexity(cls, seq: str, max_k: int = 3) -> Dict[str, float]:
        """Simple k-mer linguistic complexity for k = 1..max_k."""
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
            max_possible = min(20**k, L - k + 1)
            feats[f"lc_k{k}"] = len(observed) / max_possible if max_possible > 0 else 0.0
        return feats

    @classmethod
    def _net_charge(
        cls,
        seq: str,
        pH: float,
        include_histidine: bool = False,
    ) -> float:
        """Estimate net charge using a simple Henderson–Hasselbalch model."""
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

        # Side chains
        K = nK * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["K"])))
        R = nR * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["R"])))
        H = (
            nH * (1.0 / (1.0 + 10.0 ** (pH - PKA_SIDE["H"])))
            if include_histidine
            else 0.0
        )
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
        """Compute global descriptors for a single protein sequence."""
        s = cls._clean(seq)
        L = len(s)

        feats: Dict[str, float] = {}
        feats["length"] = float(L)
        feats.update(cls._aa_fractions(s))

        if L == 0:
            # For empty sequences, we return zeros/NaNs consistently
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
            # Optional AAIndex descriptors -> all NaN
            if aaindex_codes is not None:
                for code in aaindex_codes:
                    feats[f"aaindex_{code}_mean"] = math.nan
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
        net_charge = cls._net_charge(
            s,
            pH=pH,
            include_histidine=include_histidine_in_charge,
        )
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
                msg = (
                    "AAIndex support is not available. Make sure "
                    "`roxy.core.aaindex` (or `core.aaindex` in standalone mode) "
                    "is importable and AAIndex is configured."
                )
                logger.error("ProteinSequenceDescriptors: %s", msg)
                raise AAIndexError(msg)

            aa_feats = compute_aaindex_means_for_sequence(
                s,
                index_codes=list(aaindex_codes),
            )
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
        """Featurise all sequences in a DataFrame."""
        if sequence_col not in df.columns:
            msg = f"Column {sequence_col!r} not found in input DataFrame."
            logger.error("ProteinSequenceDescriptors.featurize_df: %s", msg)
            raise DescriptorError(msg)

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
                    msg = f"Column {id_col!r} not found in input DataFrame."
                    logger.error(
                        "ProteinSequenceDescriptors.featurize_df: %s", msg
                    )
                    raise DescriptorError(msg)
                feats[id_col] = row[id_col]
                index.append(row[id_col])
            else:
                index.append(idx)
            rows.append(feats)

        feat_df = pd.DataFrame(rows, index=index)
        if id_col is not None:
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
        """Featurise all sequences contained in a FASTA file."""
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
                    if current_id is not None:
                        ids.append(current_id)
                        seqs.append("".join(current_seq))
                        current_seq = []
                    current_id = line[1:].strip()
                else:
                    current_seq.append(line.strip())

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
        col_order = ["id", "sequence"] + [
            c for c in df.columns if c not in ("id", "sequence")
        ]
        return df[col_order]


# Backwards-compatibility alias (old name was ProteinDescriptorError)
ProteinDescriptorError = ProteinSequenceDescriptors


@dataclass
class GlobalSequenceDescriptors(BaseDescriptorEngine):
    """Roxy descriptor engine for global sequence features (with optional AAIndex).

    This class is the bridge between the low-level feature computation
    implemented in :class:`ProteinSequenceDescriptors` and the generic
    descriptor engine interface used by Roxy.

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

    name: str = "seq_global"
    pH: float = 7.0
    include_histidine_in_charge: bool = False
    aaindex_codes: Optional[Iterable[str]] = None

    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptors for all sequences in the samples table."""
        if "sequence" not in samples.columns:
            msg = (
                "GlobalSequenceDescriptors expects a 'sequence' column in "
                "`samples`."
            )
            logger.error("GlobalSequenceDescriptors.compute: %s", msg)
            raise DescriptorError(msg)

        n = samples.shape[0]
        logger.info(
            "GlobalSequenceDescriptors: computing descriptors for %d sequences "
            "(pH=%.2f, histidine_in_charge=%s, aaindex_codes=%s).",
            n,
            self.pH,
            self.include_histidine_in_charge,
            ",".join(self.aaindex_codes) if self.aaindex_codes else "None",
        )

        rows: List[Dict[str, Any]] = []
        for idx, row in samples.iterrows():
            seq = row["sequence"]
            feats = ProteinSequenceDescriptors.featurize_sequence(
                seq,
                pH=self.pH,
                include_histidine_in_charge=self.include_histidine_in_charge,
                aaindex_codes=self.aaindex_codes,
            )
            feats["_index"] = idx
            rows.append(feats)

        df = pd.DataFrame(rows).set_index("_index")

        logger.debug(
            "GlobalSequenceDescriptors: finished. Output shape=%s.",
            df.shape,
        )
        return df
