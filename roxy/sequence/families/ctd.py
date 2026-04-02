"""Composition, transition, and distribution descriptors.

This module will host CTD-style sequence descriptor implementations for
protein sequences.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return the canonical empty CTD block.
- Length-1 sequences are valid for composition features but degenerate
  for transition and distribution features.
- Sequences too short for a specific CTD subfamily should yield
  deterministic degenerate outputs rather than raising by default.
"""

__all__ = []
