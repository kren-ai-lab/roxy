"""Descriptor-family subpackage for protein sequence features.

This package hosts modular descriptor-family implementations such as
composition, grouped, AAIndex, k-mers, terminal descriptors, CTD,
patterns, order, and complexity.
"""

# Import concrete family modules directly when needed. This package does
# not re-export all families by default so the public surface stays small
# and explicit.

__all__: list[str] = []
