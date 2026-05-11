"""Normalized positional descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS
from roxy.descriptors._utils import clean_sequence
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_TRACKED_GROUPS: dict[str, frozenset[str]] = {
    "charged":    frozenset(AA_GROUPS["charged"]),
    "hydrophobic": frozenset(AA_GROUPS["hydrophobic"]),
    "aromatic":   frozenset(AA_GROUPS["aromatic"]),
    "polar":      frozenset(AA_GROUPS["polar"]),
    "positive":   frozenset(AA_GROUPS["positive"]),
    "negative":   frozenset(AA_GROUPS["negative"]),
    "gly":        frozenset("G"),
    "pro":        frozenset("P"),
    "trp":        frozenset("W"),
    "tyr":        frozenset("Y"),
}

_N_BIAS_CUTOFF = 0.33
_C_BIAS_CUTOFF = 0.67

_STAT_KEYS = (
    "count", "first_norm", "last_norm", "mean_norm", "median_norm",
    "std_norm", "span_norm", "n_bias", "c_bias", "center_mass_norm",
)


def _positional_summary(positions: list[int], seq_len: int) -> dict[str, float]:
    if not positions:
        return {
            "count": 0.0, "first_norm": _NAN, "last_norm": _NAN,
            "mean_norm": _NAN, "median_norm": _NAN, "std_norm": _NAN,
            "span_norm": _NAN, "n_bias": _NAN, "c_bias": _NAN,
            "center_mass_norm": _NAN,
        }
    norm = np.array(positions, dtype=float) / seq_len
    return {
        "count": float(len(norm)),
        "first_norm": float(norm[0]),
        "last_norm": float(norm[-1]),
        "mean_norm": float(norm.mean()),
        "median_norm": float(np.median(norm)),
        "std_norm": float(norm.std(ddof=0)),
        "span_norm": float(norm[-1] - norm[0]),
        "n_bias": float(np.mean(norm <= _N_BIAS_CUTOFF)),
        "c_bias": float(np.mean(norm >= _C_BIAS_CUTOFF)),
        "center_mass_norm": float(norm.mean()),
    }


@register("normalized", family="positional")
class NormalizedPositionDescriptor(BaseDescriptor):
    """Normalized positional statistics for tracked residue groups.

    For each group, computes count and 9 normalized position statistics
    (first/last/mean/median/std/span, N-bias, C-bias, center of mass).

    Args:
        groups: Mapping of group name → residue set. Defaults to 10 built-in
            groups (charged, hydrophobic, aromatic, polar, positive, negative,
            gly, pro, trp, tyr).

    Output columns (prefix ``normalized_``):
        ``length``, ``valid_residue_count``,
        per group: ``{name}_count``, ``{name}_{stat}`` for 9 position stats.

    """

    def __init__(
        self,
        *,
        groups: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Initialize NormalizedPositionDescriptor."""
        self.groups = groups if groups is not None else dict(_TRACKED_GROUPS)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for name in self.groups:
            feats[f"{name}_count"] = 0.0
            for stat in _STAT_KEYS[1:]:
                feats[f"{name}_{stat}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute positional features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        for name, group in self.groups.items():
            positions = [i + 1 for i, aa in enumerate(seq) if aa in group]
            summary = _positional_summary(positions, n)
            for stat, val in summary.items():
                feats[f"{name}_{stat}"] = val

        return feats
