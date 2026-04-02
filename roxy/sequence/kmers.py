"""K-mer and n-gram sequence descriptors.

This module will contain k-mer based descriptor extraction utilities
for protein sequences.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return the canonical empty descriptor
  block for the requested ``k`` values.
- Sequences shorter than ``k`` should not raise by default; they should
  return zero-valued k-mer features for that ``k``.
- Length-1 sequences are therefore valid for ``k=1`` and degenerate for
  ``k>1``.
- Unknown residues should follow the active cleaning policy before k-mer
  computation begins.
"""

__all__ = []
