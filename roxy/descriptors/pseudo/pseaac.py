"""Pseudo-amino acid composition (PseAAC) descriptor."""

from __future__ import annotations

import math

import numpy as np

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
    values: np.ndarray,
    lag: int,
) -> float:
    """Average squared-difference correlation factor across all properties."""
    n = values.shape[0]
    if n <= lag or lag < 1:
        return _NAN
    diffs = values[:-lag] - values[lag:]
    return float(np.mean(np.mean(diffs * diffs, axis=1)))


@register("pseaac", family="pseudo")
class PseAACDescriptor(BaseDescriptor):
    r"""Type I Pseudo-Amino Acid Composition (PseAAC) (Chou, 2001).

    Augments standard AAC frequencies with sequence-order correlation
    factors derived from physicochemical property differences at
    multiple lags. The correlation factor at lag :math:`\lambda` is:

    .. math::

        \theta_\lambda = \frac{1}{N - \lambda} \sum_{i=1}^{N-\lambda}
        \frac{1}{P} \sum_{p=1}^{P} (H_p(R_i) - H_p(R_{i+\lambda}))^2

    where :math:`H_p` are z-score normalized property values.
    All features are normalized so their sum equals 1:

    .. math::

        x_i = \frac{f_i}{1 + w \sum_\lambda \theta_\lambda}, \quad
        x_{20+\lambda} = \frac{w \cdot \theta_\lambda}{1 + w \sum_\lambda \theta_\lambda}

    Args:
        lam: Number of sequence-order correlation factors (lag 1..lam).
        w: Weight for correlation factors relative to AAC.
        properties: Mapping of property name to per-residue scale dict.
            Defaults to Eisenberg hydrophobicity, Hopp-Woods
            hydrophilicity, and side-chain mass.

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``pseaac_``.
        ``length``, ``valid_residue_count``,
        ``{AA}`` for each of the 20 standard AAs,
        ``theta_{1}``..``theta_{lam}``,
        ``feature_sum`` (should be :math:`\approx 1.0` for non-empty
        sequences).

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

        norm_scales = [zscore_scale(scale, _AA20_LIST) for scale in self.properties.values()]
        prop_values = np.array([[scale[aa] for scale in norm_scales] for aa in seq], dtype=float)

        thetas = []
        for lag in range(1, self.lam + 1):
            theta = _correlation_theta(prop_values, lag)
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
