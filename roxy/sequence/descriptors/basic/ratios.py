"""Ratio-oriented public surface for basic composition descriptors.

The implementation still lives in :mod:`grouped` because ratio features
depend on the same group normalization and grouped residue counting used by
``GroupedCompositionDescriptors``. Keeping the logic co-located avoids
duplicating validation/counting behavior while still giving ratios a clear
public module.
"""

from roxy.sequence.descriptors.basic.grouped import (
    GROUPED_RATIO_SPECS,
    grouped_ratio_descriptors,
)

__all__ = [
    "GROUPED_RATIO_SPECS",
    "grouped_ratio_descriptors",
]
