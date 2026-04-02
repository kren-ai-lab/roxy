"""Pattern and motif sequence descriptors.

This module will contain descriptor logic based on motifs, regex
patterns, and sequence-level pattern statistics.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return zero-valued motif counts and
  ratios where applicable.
- Length-1 sequences are valid unless a specific motif definition
  requires a longer minimum span.
- Pattern families with explicit minimum spans should document their
  degenerate output behavior for shorter sequences.
"""

__all__ = []
