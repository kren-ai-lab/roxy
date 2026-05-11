"""Family-level aggregate summary descriptors."""

from __future__ import annotations

import math
from collections import Counter
from itertools import groupby

import numpy as np

from roxy.core.constants import AA_GROUPS, CF_HELIX, CF_SHEET, CF_TURN, KD, POLARITY
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan
_LOCAL_CHARGE_WINDOW = 5

_POS = frozenset(AA_GROUPS["positive"])
_NEG = frozenset(AA_GROUPS["negative"])
_CHARGED = frozenset(AA_GROUPS["charged"])
_HYDROPHOBIC = frozenset(AA_GROUPS["hydrophobic"])
_HYDROPHILIC = frozenset(AA_GROUPS["hydrophilic"])
_POLAR = frozenset(AA_GROUPS["polar"])
_NONPOLAR = frozenset(AA_GROUPS["nonpolar"])
_DISORDER = frozenset(AA_GROUPS["disorder_promoting"])
_ORDER = frozenset(AA_GROUPS["order_promoting"])
_SULFUR = frozenset(AA_GROUPS["sulfur"])
_HYDROXYL = frozenset(AA_GROUPS["hydroxyl"])
_AMIDE = frozenset(AA_GROUPS["amide"])
_AROMATIC = frozenset(AA_GROUPS["aromatic"])

_NAN_KEYS = (
    "charge_family_mean", "charge_family_balance",
    "charge_family_local_mean", "charge_family_local_amplitude",
    "physchem_family_mean", "physchem_family_dispersion",
    "physchem_hydrophobic_balance", "physchem_polar_balance",
    "struct_family_mean", "struct_helix_sheet_balance",
    "struct_turn_secondary_balance",
    "orderdis_family_mean", "orderdis_balance",
    "functional_family_mean", "functional_reactivity_proxy",
    "complexity_family_mean", "complexity_entropy", "complexity_repeat_burden",
    "global_family_mean", "global_family_std",
    "global_family_max", "global_family_min", "global_family_amplitude",
)


def _scale_mean(seq: str, scale: dict[str, float]) -> float:
    return float(np.mean([scale[aa] for aa in seq]))


def _scale_std(seq: str, scale: dict[str, float]) -> float:
    return float(np.std([scale[aa] for aa in seq], ddof=0))


def _group_frac(seq: str, group: frozenset[str]) -> float:
    return sum(aa in group for aa in seq) / len(seq)


def _shannon_entropy(seq: str) -> float:
    counts = Counter(seq)
    probs = np.array([c / len(seq) for c in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def _longest_homo(seq: str) -> int:
    return max(len(list(g)) for _, g in groupby(seq))


def _local_charge_profile(seq: str, window: int) -> list[float]:
    ws = [seq[i : i + window] for i in range(len(seq) - window + 1)]
    return [_group_frac(w, _POS) - _group_frac(w, _NEG) for w in ws]


def _charge_feats(seq: str, n: int) -> tuple[float, dict[str, float]]:
    pos_frac = _group_frac(seq, _POS)
    neg_frac = _group_frac(seq, _NEG)
    charged_frac = _group_frac(seq, _CHARGED)
    charge_mean = float(np.mean([pos_frac, neg_frac, charged_frac]))

    if n >= _LOCAL_CHARGE_WINDOW:
        profile = _local_charge_profile(seq, _LOCAL_CHARGE_WINDOW)
        local_mean = float(np.nanmean(profile))
        local_amp = float(np.nanmax(profile) - np.nanmin(profile))
    else:
        local_mean, local_amp = _NAN, _NAN

    return pos_frac, neg_frac, {
        "charge_family_mean": charge_mean,
        "charge_family_balance": pos_frac - neg_frac,
        "charge_family_local_mean": local_mean,
        "charge_family_local_amplitude": local_amp,
    }


def _physchem_feats(seq: str) -> dict[str, float]:
    hydropathy_mean = _scale_mean(seq, KD)
    polarity_mean = _scale_mean(seq, POLARITY)
    return {
        "physchem_family_mean": float(np.mean([hydropathy_mean, polarity_mean])),
        "physchem_family_dispersion": float(np.mean([_scale_std(seq, KD), _scale_std(seq, POLARITY)])),
        "physchem_hydrophobic_balance": _group_frac(seq, _HYDROPHOBIC) - _group_frac(seq, _HYDROPHILIC),
        "physchem_polar_balance": _group_frac(seq, _POLAR) - _group_frac(seq, _NONPOLAR),
    }


def _struct_feats(seq: str) -> dict[str, float]:
    helix = _scale_mean(seq, CF_HELIX)
    sheet = _scale_mean(seq, CF_SHEET)
    turn = _scale_mean(seq, CF_TURN)
    return {
        "struct_family_mean": float(np.mean([helix, sheet, turn])),
        "struct_helix_sheet_balance": helix - sheet,
        "struct_turn_secondary_balance": turn - (helix + sheet) / 2.0,
    }


def _orderdis_feats(seq: str) -> dict[str, float]:
    dis = _group_frac(seq, _DISORDER)
    ord_ = _group_frac(seq, _ORDER)
    return {
        "orderdis_family_mean": float(np.mean([dis, ord_])),
        "orderdis_balance": dis - ord_,
    }


def _functional_feats(seq: str, pos_frac: float, neg_frac: float) -> dict[str, float]:
    sulfur = _group_frac(seq, _SULFUR)
    hydroxyl = _group_frac(seq, _HYDROXYL)
    amide = _group_frac(seq, _AMIDE)
    aromatic = _group_frac(seq, _AROMATIC)
    return {
        "functional_family_mean": float(np.mean([sulfur, hydroxyl, amide, aromatic])),
        "functional_reactivity_proxy": float(np.mean([sulfur, hydroxyl, pos_frac, neg_frac])),
    }


def _complexity_feats(seq: str, n: int) -> dict[str, float]:
    entropy = _shannon_entropy(seq)
    burden = _longest_homo(seq) / n
    return {
        "complexity_family_mean": float(np.mean([entropy, burden])),
        "complexity_entropy": entropy,
        "complexity_repeat_burden": burden,
    }


@register("family_summary", family="hybrid")
class FamilySummaryDescriptor(BaseDescriptor):
    """Family-level aggregate summary descriptors.

    Computes 6 family-level summaries (charge, physicochemical, structural
    propensity, order/disorder, functional residue, complexity) and 5 global
    meta-summaries across all families.

    Output columns (prefix ``family_summary_``):
        ``length``, ``valid_residue_count``,
        4 charge family, 4 physicochemical family, 3 structural propensity,
        2 order/disorder, 2 functional, 3 complexity, 5 global summaries.
        Total: 2 + 4 + 4 + 3 + 2 + 2 + 3 + 5 = 25 columns.
    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute family summary features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {"length": float(n), "valid_residue_count": float(n)}

        if n == 0:
            feats.update(dict.fromkeys(_NAN_KEYS, _NAN))
            return feats

        pos_frac, neg_frac, charge = _charge_feats(seq, n)
        physchem = _physchem_feats(seq)
        struct = _struct_feats(seq)
        orderdis = _orderdis_feats(seq)
        functional = _functional_feats(seq, pos_frac, neg_frac)
        complexity = _complexity_feats(seq, n)

        feats.update(charge)
        feats.update(physchem)
        feats.update(struct)
        feats.update(orderdis)
        feats.update(functional)
        feats.update(complexity)

        family_core = [
            charge["charge_family_mean"],
            physchem["physchem_family_mean"],
            struct["struct_family_mean"],
            orderdis["orderdis_family_mean"],
            functional["functional_family_mean"],
            complexity["complexity_family_mean"],
        ]

        f_max = float(np.nanmax(family_core))
        f_min = float(np.nanmin(family_core))
        feats["global_family_mean"] = float(np.nanmean(family_core))
        feats["global_family_std"] = float(np.nanstd(family_core, ddof=0))
        feats["global_family_max"] = f_max
        feats["global_family_min"] = f_min
        feats["global_family_amplitude"] = f_max - f_min

        return feats
