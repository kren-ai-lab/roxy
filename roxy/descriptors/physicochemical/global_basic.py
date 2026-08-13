"""Global basic sequence descriptor."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from roxy.core.constants import (
    AA_GROUPS,
    AA_MOLECULAR_WEIGHT,
    ACCEPTORS,
    BOMAN,
    CF_HELIX,
    CF_SHEET,
    CF_TURN,
    DONORS,
    FLEXIBILITY,
    KD,
    POLARITY,
)
from roxy.descriptors._utils import (
    fraction_from_group,
    linguistic_complexity,
    longest_homopolymer_run,
    net_charge_at_ph,
    scale_values,
    shannon_entropy,
)
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_WATER_MW = 18.015


def _aliphatic_index(seq: str, n: int) -> float:
    r"""Aliphatic index (Ikai, 1980).

    .. math::

        AI = 100 \times (x_A + 2.9 \cdot x_V + 3.9 \cdot (x_I + x_L))

    where :math:`x` are mole fractions.
    """
    c = Counter(seq)
    return 100 * (c["A"] / n + 2.9 * c["V"] / n + 3.9 * (c["I"] + c["L"]) / n)


def _repeated_dipeptide_fraction(seq: str) -> float:
    """Fraction of dipeptide occurrences that belong to repeated dipeptides."""
    if len(seq) < 2:  # noqa: PLR2004
        return _NAN
    kmers = [seq[i : i + 2] for i in range(len(seq) - 1)]
    counts = Counter(kmers)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(kmers)


@register("global_basic", family="physicochemical")
class GlobalBasicDescriptor(BaseDescriptor):
    r"""44 global sequence properties: composition, scales, charge, complexity.

    Includes:

    * **Molecular weight**:

      .. math:: MW = \sum MW_i - (N - 1) \times 18.015

    * **Aliphatic index** (Ikai, 1980):

      .. math:: AI = 100 \times (x_A + 2.9 x_V + 3.9 (x_I + x_L))
    * **Scale statistics**: mean/std of Kyte-Doolittle, Zimmerman polarity,
      and Bhaskaran-Ponnuswamy flexibility.
    * **Chou-Fasman propensity means** (helix, sheet, turn) and the
      **Boman index**.
    * **Charge**: net charge at pH 7 (Henderson-Hasselbalch), FCR, NCPR.
    * **Complexity**: Shannon entropy, linguistic complexity (k=1,2,3),
      longest homopolymer run.

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``global_basic_``.
        ``length``, ``valid_residue_count``, ``unique_residue_count``,
        ``molecular_weight``, 15 group fractions, ``aliphatic_index``,
        ``hydropathy_mean/std``, ``polarity_mean/std``,
        ``flexibility_mean/std``,
        ``helix/sheet/turn_propensity_mean``, ``boman_index``,
        ``net_charge_ph7``, ``fcr``, ``ncpr``,
        ``acidic_basic_ratio``, ``basic_acidic_ratio``,
        ``donors_per_residue``, ``acceptors_per_residue``,
        ``shannon_entropy``, ``linguistic_complexity_k1/k2/k3``,
        ``longest_homopolymer_run``, ``repeated_dipeptide_fraction``,
        ``local_hydropathy_amplitude_w5``.

    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute global basic features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)
        empty = n == 0
        counts = Counter(seq)

        feats["unique_residue_count"] = float(len(counts)) if not empty else 0.0

        if empty:
            for key in (
                "molecular_weight",
                "aliphatic_index",
                "hydropathy_mean",
                "hydropathy_std",
                "polarity_mean",
                "polarity_std",
                "flexibility_mean",
                "flexibility_std",
                "helix_propensity_mean",
                "sheet_propensity_mean",
                "turn_propensity_mean",
                "boman_index",
                "net_charge_ph7",
                "fcr",
                "ncpr",
                "acidic_basic_ratio",
                "basic_acidic_ratio",
                "donors_per_residue",
                "acceptors_per_residue",
                "shannon_entropy",
                "linguistic_complexity_k1",
                "linguistic_complexity_k2",
                "linguistic_complexity_k3",
                "repeated_dipeptide_fraction",
                "local_hydropathy_amplitude_w5",
            ):
                feats[key] = _NAN
            for group in AA_GROUPS:
                feats[f"{group}_fraction"] = _NAN
            feats["longest_homopolymer_run"] = 0.0
            return feats

        # MW: sum of residue weights minus water per peptide bond
        feats["molecular_weight"] = sum(AA_MOLECULAR_WEIGHT[aa] for aa in seq) - (n - 1) * _WATER_MW

        # Group fractions
        for group, members in AA_GROUPS.items():
            feats[f"{group}_fraction"] = fraction_from_group(seq, members)

        feats["aliphatic_index"] = _aliphatic_index(seq, n)

        # Scale means/stds
        for name, scale in (
            ("hydropathy", KD),
            ("polarity", POLARITY),
            ("flexibility", FLEXIBILITY),
        ):
            vals = scale_values(seq, scale)
            feats[f"{name}_mean"] = float(np.mean(vals)) if vals else _NAN
            feats[f"{name}_std"] = float(np.std(vals, ddof=0)) if vals else _NAN

        for name, scale in (
            ("helix_propensity", CF_HELIX),
            ("sheet_propensity", CF_SHEET),
            ("turn_propensity", CF_TURN),
        ):
            vals = scale_values(seq, scale)
            feats[f"{name}_mean"] = float(np.mean(vals)) if vals else _NAN

        # Boman index: negated mean of the residue solubility values.
        boman_vals = scale_values(seq, BOMAN)
        feats["boman_index"] = -float(np.mean(boman_vals)) if boman_vals else _NAN

        # Charge
        feats["net_charge_ph7"] = net_charge_at_ph(seq, 7.0)
        pos = fraction_from_group(seq, AA_GROUPS["positive"])
        neg = fraction_from_group(seq, AA_GROUPS["negative"])
        feats["fcr"] = pos + neg
        feats["ncpr"] = pos - neg
        n_acidic = sum(aa in AA_GROUPS["negative"] for aa in seq)
        n_basic = sum(aa in AA_GROUPS["positive"] for aa in seq)
        feats["acidic_basic_ratio"] = n_acidic / n_basic if n_basic else _NAN
        feats["basic_acidic_ratio"] = n_basic / n_acidic if n_acidic else _NAN

        # Fraction of residues able to donate / accept a side-chain H-bond
        feats["donors_per_residue"] = fraction_from_group(seq, DONORS)
        feats["acceptors_per_residue"] = fraction_from_group(seq, ACCEPTORS)

        # Complexity
        feats["shannon_entropy"] = shannon_entropy(seq)
        feats["linguistic_complexity_k1"] = linguistic_complexity(seq, 1)
        feats["linguistic_complexity_k2"] = linguistic_complexity(seq, 2)
        feats["linguistic_complexity_k3"] = linguistic_complexity(seq, 3)
        feats["longest_homopolymer_run"] = float(longest_homopolymer_run(seq))
        feats["repeated_dipeptide_fraction"] = _repeated_dipeptide_fraction(seq)

        # Local hydropathy amplitude (window=5)
        kd_vals = scale_values(seq, KD)
        if len(kd_vals) >= 5:  # noqa: PLR2004
            w_means = [float(np.mean(kd_vals[i : i + 5])) for i in range(len(kd_vals) - 4)]
            feats["local_hydropathy_amplitude_w5"] = max(w_means) - min(w_means)
        else:
            feats["local_hydropathy_amplitude_w5"] = _NAN

        return feats
