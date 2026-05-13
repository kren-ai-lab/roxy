"""Pseudo-amino acid composition (PseAAC) descriptor."""

from __future__ import annotations

import math

from roxy.core.constants import AA20, EISENBERG, HYDROPHILICITY, SIDECHAIN_MASS
from roxy.descriptors._utils import zscore_scale
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_AA20_LIST: list[str] = sorted(AA20)

_DEFAULT_PROPERTIES: dict[str, dict[str, float]] = {
    "hydrophobicity": EISENBERG,
    "hydrophilicity": HYDROPHILICITY,
    "sidechain_mass": SIDECHAIN_MASS,
}


def _correlation_theta(
    seq: str,
    lag: int,
    norm_scales: dict[str, dict[str, float]],
) -> float:
    """Average squared-difference correlation factor across all properties."""
    n = len(seq)
    if n <= lag or lag < 1:
        return _NAN
    total = 0.0
    for i in range(n - lag):
        aa1, aa2 = seq[i], seq[i + lag]
        diffs_sq = [(s[aa1] - s[aa2]) ** 2 for s in norm_scales.values()]
        total += sum(diffs_sq) / len(diffs_sq)
    return total / (n - lag)


@register("pseaac", family="pseudo")
class PseAACDescriptor(BaseDescriptor):
    """Type I Pseudo-Amino Acid Composition (PseAAC).

    Augments standard AAC frequencies with sequence-order correlation
    factors (theta) derived from physicochemical property differences
    at multiple lags. All 20 AA features and lambda theta features
    are normalized so that their sum equals 1.

    Args:
        lam: Number of sequence-order correlation factors (lag 1..lam).
        w: Weight for correlation factors relative to AAC.
        properties: Mapping of property name → per-residue scale dict.
            Defaults to Eisenberg hydrophobicity, Hopp-Woods hydrophilicity,
            and side-chain mass.

    Output columns (prefix ``pseaac_``):
        ``length``, ``valid_residue_count``,
        ``{AA}`` for each of the 20 standard AAs,
        ``theta_{1}``..``theta_{lam}``,
        ``feature_sum`` (should be ≈ 1.0 for non-empty sequences).

    """

    def __init__(
        self,
        *,
        lam: int = 5,
        w: float = 0.05,
        properties: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize PseAACDescriptor."""
        self.lam = lam
        self.w = w
        self.properties = properties if properties is not None else dict(_DEFAULT_PROPERTIES)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for aa in _AA20_LIST:
            feats[aa] = _NAN
        for i in range(1, self.lam + 1):
            feats[f"theta_{i}"] = _NAN
        feats["feature_sum"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute PseAAC features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        norm_scales = {name: zscore_scale(scale, _AA20_LIST) for name, scale in self.properties.items()}

        thetas = []
        for lag in range(1, self.lam + 1):
            theta = _correlation_theta(seq, lag, norm_scales)
            thetas.append(0.0 if math.isnan(theta) else theta)

        denom = 1.0 + self.w * sum(thetas)

        aac_counts = {aa: seq.count(aa) for aa in _AA20_LIST}
        for aa in _AA20_LIST:
            feats[aa] = (aac_counts[aa] / n) / denom

        for i, theta in enumerate(thetas, start=1):
            feats[f"theta_{i}"] = (self.w * theta) / denom

        feats["feature_sum"] = sum(feats[aa] for aa in _AA20_LIST) + sum(
            feats[f"theta_{i}"] for i in range(1, self.lam + 1)
        )

        return feats
