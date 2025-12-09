"""
Descriptor engines for Roxy.

This subpackage groups descriptor engines for different object types:

- sequences  → :class:`GlobalSequenceDescriptors`
- structures → :class:`BasicStructureDescriptors`
- molecules  → :class:`BasicMoleculeDescriptors`

It also exposes:

- :data:`DESCRIPTOR_REGISTRY`  → the shared registry of engines
- :func:`compute_descriptors`  → a convenience helper for applying
  engines to a :class:`~roxy.core.dataset.RoxyDataset`.
"""

from __future__ import annotations

from .sequences import GlobalSequenceDescriptors
from .sequences import ProteinDescriptorError
from .structures import BasicStructureDescriptors
from .compounds import BasicMoleculeDescriptors
from .registry import DESCRIPTOR_REGISTRY
from .helpers import compute_descriptors

__all__ = [
    # Engines
    "GlobalSequenceDescriptors",
    "BasicStructureDescriptors",
    "BasicMoleculeDescriptors",
    "ProteinDescriptorError",
    # Registry
    "DESCRIPTOR_REGISTRY",
    # Helpers
    "compute_descriptors",
]
