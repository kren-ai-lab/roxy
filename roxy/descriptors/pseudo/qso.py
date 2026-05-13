"""Quasi-sequence-order (QSO) descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA20, EISENBERG, HYDROPHILICITY, SIDECHAIN_MASS
from roxy.descriptors._utils import zscore_scale
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_AA20 = sorted(AA20)

_DEFAULT_PROPERTIES = {
    "hydrophobicity": EISENBERG,
    "hydrophilicity": HYDROPHILICITY,
    "sidechain_mass": SIDECHAIN_MASS,
}


def _residue_dist(aa1: str, aa2: str, norm_scales: list[dict[str, float]]) -> float:
    return float(np.mean([(s[aa1] - s[aa2]) ** 2 for s in norm_scales]))


@register("qso", family="pseudo")
class QSODescriptor(BaseDescriptor):
    """Quasi-sequence-order descriptors (Chou 2001).

    Extends AAC with sequence-order coupling factors (tau) that capture
    long-range residue correlations via physicochemical property distances.

    Args:
        lam: Maximum lag (number of coupling factors). Default ``5``.
        w: Weight of coupling factors relative to composition. Default ``0.1``.
        properties: Mapping of property name → scale dict. Defaults to
            Eisenberg hydrophobicity, Hopp-Woods hydrophilicity, side-chain mass.

    Output columns (prefix ``qso_``):
        ``length``, ``valid_residue_count``,
        20 quasi-composition columns ``{AA}``,
        ``lam`` coupling columns ``tau_1`` … ``tau_{lam}``,
        ``feature_sum`` (≈ 1.0 for non-empty sequences).
        Total: 2 + 20 + lam + 1 = 28 at default lam=5.

    """

    def __init__(
        self,
        *,
        lam: int = 5,
        w: float = 0.1,
        properties: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize QSODescriptor."""
        self.lam = lam
        self.w = w
        self.properties = properties if properties is not None else dict(_DEFAULT_PROPERTIES)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for aa in _AA20:
            feats[aa] = _NAN
        for i in range(1, self.lam + 1):
            feats[f"tau_{i}"] = _NAN
        feats["feature_sum"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute QSO features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        norm_scales = [zscore_scale(s, _AA20) for s in self.properties.values()]

        aac = {aa: seq.count(aa) / n for aa in _AA20}

        couplings: list[float] = []
        for lag in range(1, self.lam + 1):
            if n <= lag:
                couplings.append(0.0)
            else:
                vals = [_residue_dist(seq[i], seq[i + lag], norm_scales) for i in range(n - lag)]
                couplings.append(float(np.mean(vals)))

        denom = 1.0 + self.w * sum(couplings)

        for aa in _AA20:
            feats[aa] = aac[aa] / denom

        for i, cf in enumerate(couplings, start=1):
            feats[f"tau_{i}"] = (self.w * cf) / denom

        feats["feature_sum"] = sum(feats[aa] for aa in _AA20) + sum(
            feats[f"tau_{i}"] for i in range(1, self.lam + 1)
        )

        return feats
