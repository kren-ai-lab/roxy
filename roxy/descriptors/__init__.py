
"""Descriptor engines and helpers for Roxy."""
from __future__ import annotations

from typing import Iterable

from roxy.core.dataset import RoxyDataset
from .registry import DESCRIPTOR_REGISTRY


def compute_descriptors(
    dataset: RoxyDataset,
    engines: Iterable[str],
    feature_key_prefix: str = "",
) -> RoxyDataset:
    """Compute and attach descriptor tables to a RoxyDataset."""
    for name in engines:
        engine = DESCRIPTOR_REGISTRY[name]
        X = engine.compute(dataset.samples)
        dataset.add_features(f"{feature_key_prefix}{name}", X)
    return dataset
