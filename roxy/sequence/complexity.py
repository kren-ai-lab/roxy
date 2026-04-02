"""Sequence complexity descriptors.

This module will host descriptors related to repetition, entropy, and
other complexity-oriented sequence measurements.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return the canonical empty complexity
  block for the chosen features.
- Length-1 sequences are valid and should yield deterministic scalar
  outputs.
- Families with an internal scale such as ``k`` or window size should
  document their degenerate behavior when the sequence is shorter than
  that scale.
"""

__all__ = []
