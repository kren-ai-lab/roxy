"""Charge and protonation-state descriptors."""

from __future__ import annotations

import math

from roxy.core.constants import PKA_C_TERM, PKA_N_TERM, PKA_SIDE
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

from ._utils import profile_stats, terminal_segment, windows

_NAN = math.nan

_POSITIVE_GROUP = frozenset("KRH")
_NEGATIVE_GROUP = frozenset("DECY")
_ACIDIC_GROUP = frozenset("DE")
_BASIC_GROUP = frozenset("KRH")
_IONIZABLE_GROUP = frozenset("CDEHKRY")
_POSITIVE_IONIZABLE = frozenset("KRH")
_NEGATIVE_IONIZABLE = frozenset("DECY")


def _net_charge(seq: str, ph: float) -> float:
    """Compute Henderson-Hasselbalch net charge including termini."""
    if not seq:
        return _NAN
    pos = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))
    neg = 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in _POSITIVE_IONIZABLE:
            pos += 1.0 / (1.0 + 10 ** (ph - pka))
        elif aa in _NEGATIVE_IONIZABLE:
            neg += 1.0 / (1.0 + 10 ** (pka - ph))
    return pos - neg


def _protonated_basic_fraction(seq: str, ph: float) -> float:
    """Mean protonation degree of basic residues (KRH)."""
    vals = [
        1.0 / (1.0 + 10 ** (ph - PKA_SIDE[aa]))
        for aa in seq
        if aa in _BASIC_GROUP and aa in PKA_SIDE
    ]
    return float(sum(vals) / len(vals)) if vals else _NAN


def _deprotonated_acidic_fraction(seq: str, ph: float) -> float:
    """Mean deprotonation degree of acidic residues (DE)."""
    vals = [
        1.0 / (1.0 + 10 ** (PKA_SIDE[aa] - ph))
        for aa in seq
        if aa in _ACIDIC_GROUP and aa in PKA_SIDE
    ]
    return float(sum(vals) / len(vals)) if vals else _NAN


@register("charge", family="physicochemical")
class ChargeDescriptor(BaseDescriptor):
    """Charge fractions, net charge, and local charge profiles.

    Computes fraction-based global charge statistics and pH-dependent
    properties at configurable pH values, including windowed local
    charge density and terminal charge asymmetry.

    Args:
        ph_values: Tuple of pH values for ionisation-state calculations.
        local_window: Window size for local charge density profile.
        terminal_window: Residue count for N/C-terminal segments.

    Output columns (prefix ``charge_``):
        ``length``, ``valid_residue_count``,
        ``positive/negative/ionizable_fraction``,
        ``fcr``, ``ncpr``, ``basic_acidic_ratio``, ``acidic_basic_ratio``,
        per-pH: ``net_charge/density/protonated_basic_fraction/
        deprotonated_acidic_fraction/local_{stat}/
        nterm_net/cterm_net/nterm_density/cterm_density/terminal_asymmetry``.

    """

    def __init__(
        self,
        *,
        ph_values: tuple[float, ...] = (5.0, 7.0, 9.0),
        local_window: int = 5,
        terminal_window: int = 10,
    ) -> None:
        """Initialize ChargeDescriptor."""
        self.ph_values = tuple(ph_values)
        self.local_window = local_window
        self.terminal_window = terminal_window

    def _ph_tag(self, ph: float) -> str:
        return str(ph).replace(".", "p")

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for key in (
            "positive_fraction", "negative_fraction", "ionizable_fraction",
            "fcr", "ncpr", "basic_acidic_ratio", "acidic_basic_ratio",
        ):
            feats[key] = _NAN
        for ph in self.ph_values:
            tag = self._ph_tag(ph)
            for key in (
                f"net_charge_ph{tag}", f"density_ph{tag}",
                f"protonated_basic_fraction_ph{tag}",
                f"deprotonated_acidic_fraction_ph{tag}",
                f"local_mean_ph{tag}", f"local_std_ph{tag}",
                f"local_min_ph{tag}", f"local_max_ph{tag}",
                f"local_amplitude_ph{tag}", f"local_start_end_diff_ph{tag}",
                f"nterm_net_ph{tag}", f"cterm_net_ph{tag}",
                f"nterm_density_ph{tag}", f"cterm_density_ph{tag}",
                f"terminal_asymmetry_ph{tag}",
            ):
                feats[key] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute charge features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        pos_frac = sum(aa in _POSITIVE_GROUP for aa in seq) / n
        neg_frac = sum(aa in _NEGATIVE_GROUP for aa in seq) / n
        acidic_n = sum(aa in _ACIDIC_GROUP for aa in seq)
        basic_n = sum(aa in _BASIC_GROUP for aa in seq)

        feats["positive_fraction"] = pos_frac
        feats["negative_fraction"] = neg_frac
        feats["ionizable_fraction"] = sum(aa in _IONIZABLE_GROUP for aa in seq) / n
        feats["fcr"] = pos_frac + neg_frac
        feats["ncpr"] = pos_frac - neg_frac
        feats["basic_acidic_ratio"] = basic_n / acidic_n if acidic_n else _NAN
        feats["acidic_basic_ratio"] = acidic_n / basic_n if basic_n else _NAN

        nterm = terminal_segment(seq, "N", self.terminal_window)
        cterm = terminal_segment(seq, "C", self.terminal_window)

        for ph in self.ph_values:
            tag = self._ph_tag(ph)
            nc = _net_charge(seq, ph)
            feats[f"net_charge_ph{tag}"] = nc
            feats[f"density_ph{tag}"] = nc / n
            feats[f"protonated_basic_fraction_ph{tag}"] = _protonated_basic_fraction(seq, ph)
            feats[f"deprotonated_acidic_fraction_ph{tag}"] = _deprotonated_acidic_fraction(seq, ph)

            local_profile = [
                _net_charge(w, ph) / self.local_window
                for w in windows(seq, self.local_window)
            ]
            stats = profile_stats(local_profile)
            feats[f"local_mean_ph{tag}"] = stats["mean"]
            feats[f"local_std_ph{tag}"] = stats["std"]
            feats[f"local_min_ph{tag}"] = stats["min"]
            feats[f"local_max_ph{tag}"] = stats["max"]
            feats[f"local_amplitude_ph{tag}"] = stats["amplitude"]
            feats[f"local_start_end_diff_ph{tag}"] = stats["start_end_diff"]

            nterm_nc = _net_charge(nterm, ph)
            cterm_nc = _net_charge(cterm, ph)
            nterm_n = len(nterm) or 1
            cterm_n = len(cterm) or 1
            feats[f"nterm_net_ph{tag}"] = nterm_nc
            feats[f"cterm_net_ph{tag}"] = cterm_nc
            feats[f"nterm_density_ph{tag}"] = nterm_nc / nterm_n
            feats[f"cterm_density_ph{tag}"] = cterm_nc / cterm_n
            feats[f"terminal_asymmetry_ph{tag}"] = (
                feats[f"nterm_density_ph{tag}"] - feats[f"cterm_density_ph{tag}"]
            )

        return feats
