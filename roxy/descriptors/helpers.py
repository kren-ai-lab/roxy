"""Helper utilities for working with descriptor engines.

This module currently provides:

- ``compute_descriptors``: apply one or more registered descriptor
  engines to a :class:`roxy.core.dataset.RoxyDataset` and attach the
  resulting feature blocks.
"""

from __future__ import annotations

from typing import Iterable

from roxy.core.dataset import RoxyDataset
from .registry import DESCRIPTOR_REGISTRY


def compute_descriptors(
    dataset: RoxyDataset,
    engines: Iterable[str],
    *,
    feature_key_prefix: str = "",
) -> RoxyDataset:
    """
    Compute one or multiple descriptor engines and attach the resulting
    feature blocks to the given :class:`RoxyDataset`.

    Parameters
    ----------
    dataset :
        The dataset to enrich with descriptor feature blocks.
    engines :
        Iterable of descriptor names registered in
        :data:`DESCRIPTOR_REGISTRY`, e.g.::

            ["seq_global", "struct_basic", "mol_basic"]

    feature_key_prefix :
        Optional prefix added to the feature-key when inserting the
        feature block into the dataset. This is useful when calling the
        same engine multiple times or producing multiple variants.

    Returns
    -------
    RoxyDataset
        The same dataset instance, enriched with descriptor blocks.

    Raises
    ------
    KeyError
        If any requested engine name is not present in
        :data:`DESCRIPTOR_REGISTRY`.

    Examples
    --------
    >>> ds = RoxyDataset(samples=df, name="toy")
    >>> compute_descriptors(ds, ["seq_global"])
    >>> ds.feature_blocks
    ['seq_global']
    """
    for name in engines:
        if name not in DESCRIPTOR_REGISTRY:
            raise KeyError(
                f"Descriptor engine '{name}' not found. "
                f"Available engines: {', '.join(DESCRIPTOR_REGISTRY.keys())}"
            )

        engine = DESCRIPTOR_REGISTRY[name]
        X_desc = engine.compute(dataset.samples)

        feature_key = f"{feature_key_prefix}{name}"
        dataset.add_features(feature_key, X_desc)

    return dataset
