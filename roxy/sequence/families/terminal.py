"""Terminal-region sequence descriptors.

This module is reserved for N-terminal and C-terminal descriptor logic
for protein sequences.

Planned edge-case rule set
--------------------------
- Empty cleaned sequences should return the canonical empty descriptor
  block for the selected terminal features.
- If a sequence is shorter than the requested terminal window, the full
  cleaned sequence should be used rather than raising by default.
- Unknown residues should follow the active cleaning policy before
  terminal extraction begins.
"""

__all__ = []
