"""Physicochemical autocorrelation descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import FLEXIBILITY, KD, POLARITY, VOLUME
from roxy.descriptors._utils import clean_sequence
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_CHARGE_PROXY: dict[str, float] = {
    "A": 0.0,
    "C": 0.0,
    "D": -1.0,
    "E": -1.0,
    "F": 0.0,
    "G": 0.0,
    "H": 0.5,
    "I": 0.0,
    "K": 1.0,
    "L": 0.0,
    "M": 0.0,
    "N": 0.0,
    "P": 0.0,
    "Q": 0.0,
    "R": 1.0,
    "S": 0.0,
    "T": 0.0,
    "V": 0.0,
    "W": 0.0,
    "Y": 0.0,
}

_DEFAULT_SCALES: dict[str, dict[str, float]] = {
    "hydrophobicity": KD,
    "polarity": POLARITY,
    "flexibility": FLEXIBILITY,
    "volume": VOLUME,
    "charge_proxy": _CHARGE_PROXY,
}


def _moreau_broto(values: np.ndarray, lag: int) -> float:
    """Unnormalized Moreau-Broto autocorrelation at given lag."""
    n = len(values)
    if n <= lag or lag < 1:
        return _NAN
    return float(np.sum(values[:-lag] * values[lag:]) / (n - lag))


def _moran(values: np.ndarray, lag: int) -> float:
    """Moran autocorrelation at given lag."""
    n = len(values)
    if n <= lag or lag < 1:
        return _NAN
    mean_val = np.mean(values)
    denom = np.mean((values - mean_val) ** 2)
    if denom == 0:
        return _NAN
    numer = np.sum((values[:-lag] - mean_val) * (values[lag:] - mean_val)) / (n - lag)
    return float(numer / denom)


def _geary(values: np.ndarray, lag: int) -> float:
    """Geary autocorrelation at given lag."""
    n = len(values)
    if n <= lag or lag < 1:
        return _NAN
    mean_val = np.mean(values)
    denom = np.mean((values - mean_val) ** 2)
    if denom == 0:
        return _NAN
    numer = np.sum((values[:-lag] - values[lag:]) ** 2) / (2 * (n - lag))
    return float(numer / denom)


@register("autocorrelation", family="autocorrelation")
class AutocorrelationDescriptor(BaseDescriptor):
    """Moreau-Broto, Moran, and Geary autocorrelation over physicochemical scales.

    Computes three types of sequence autocorrelation at multiple lags
    for each configured physicochemical scale.

    Args:
        scales: Mapping of scale name → per-residue value dict. Defaults to
            5 built-in scales (hydrophobicity, polarity, flexibility, volume,
            charge_proxy).
        lags: Sequence of lag values to compute.

    Output columns (prefix ``autocorrelation_``):
        ``length``, ``valid_residue_count``,
        ``mb_{scale}_lag{k}``, ``moran_{scale}_lag{k}``, ``geary_{scale}_lag{k}``
        for each scale and lag.

    """

    def __init__(
        self,
        *,
        scales: dict[str, dict[str, float]] | None = None,
        lags: tuple[int, ...] = (1, 2, 3, 4, 5),
    ) -> None:
        """Initialize AutocorrelationDescriptor."""
        self.scales = scales if scales is not None else dict(_DEFAULT_SCALES)
        self.lags = tuple(lags)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for scale_name in self.scales:
            for lag in self.lags:
                feats[f"mb_{scale_name}_lag{lag}"] = _NAN
                feats[f"moran_{scale_name}_lag{lag}"] = _NAN
                feats[f"geary_{scale_name}_lag{lag}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute autocorrelation features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        for scale_name, scale in self.scales.items():
            values = np.array([scale[aa] for aa in seq if aa in scale], dtype=float)
            for lag in self.lags:
                feats[f"mb_{scale_name}_lag{lag}"] = _moreau_broto(values, lag)
                feats[f"moran_{scale_name}_lag{lag}"] = _moran(values, lag)
                feats[f"geary_{scale_name}_lag{lag}"] = _geary(values, lag)

        return feats
