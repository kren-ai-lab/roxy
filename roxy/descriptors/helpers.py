"""Helper utilities for working with descriptor engines.

This module currently provides:

- :func:`compute_descriptors` – apply one or more registered descriptor
  engines to a :class:`roxy.core.dataset.RoxyDataset` and attach the
  resulting feature blocks.
"""

from __future__ import annotations

from typing import Iterable, List

from roxy.core.dataset import RoxyDataset
from roxy.core.exceptions import DescriptorError
from roxy.core.logging_utils import get_logger

from .registry import DESCRIPTOR_REGISTRY

logger = get_logger(__name__)


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

            ["seq_global"]

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
    DescriptorError
        If any requested engine name is not present in
        :data:`DESCRIPTOR_REGISTRY`.
    """
    engine_names: List[str] = list(engines)
    logger.info(
        "compute_descriptors: running engines [%s] on dataset %r.",
        ", ".join(engine_names),
        dataset.name,
    )

    for name in engine_names:
        if name not in DESCRIPTOR_REGISTRY:
            msg = (
                f"Descriptor engine '{name}' not found. "
                f"Available engines: {', '.join(DESCRIPTOR_REGISTRY.keys())}"
            )
            logger.error("compute_descriptors: %s", msg)
            raise DescriptorError(msg)

        engine = DESCRIPTOR_REGISTRY[name]
        logger.debug("compute_descriptors: computing engine '%s'.", name)
        X_desc = engine.compute(dataset.samples)

        feature_key = f"{feature_key_prefix}{name}"
        dataset.add_features(feature_key, X_desc)
        logger.info(
            "compute_descriptors: attached feature block '%s' with shape %s.",
            feature_key,
            X_desc.shape,
        )

    return dataset
