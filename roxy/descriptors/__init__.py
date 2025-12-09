"""Descriptor engines for Roxy (sequence-focused).

This subpackage currently groups **sequence-based** descriptor engines:

- :class:`GlobalSequenceDescriptors` – global protein sequence descriptors.
- :class:`ProteinSequenceDescriptors` – low-level helpers usable on their own.

It also exposes:

- :data:`DESCRIPTOR_REGISTRY` – the shared registry of engines.
- :func:`compute_descriptors` – convenience helper for applying engines
  to a :class:`~roxy.core.dataset.RoxyDataset`.
"""

from __future__ import annotations

from .sequences import GlobalSequenceDescriptors, ProteinSequenceDescriptors, ProteinDescriptorError
from .registry import DESCRIPTOR_REGISTRY
from .helpers import compute_descriptors

__all__ = [
    # Engines / helpers
    "GlobalSequenceDescriptors",
    "ProteinSequenceDescriptors",
    # Backwards-compatibility alias
    "ProteinDescriptorError",
    # Registry + helper
    "DESCRIPTOR_REGISTRY",
    "compute_descriptors",
]
