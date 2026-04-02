"""Sequence-order descriptors.

This module is reserved for descriptors that capture residue order,
autocorrelation, or other position-aware sequence properties.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return the canonical empty descriptor
  block.
- Length-1 sequences are valid but often degenerate for lag-based or
  autocorrelation-style summaries.
- Minimum-length requirements should yield deterministic empty or zero
  outputs unless a specific method explicitly requires strict failure.
"""

__all__ = []
