"""AAIndex-backed descriptor family exports."""

from roxy.sequence.descriptors.aaindex.aaindex import (
    AAIndexDescriptors,
    aaindex_mean_descriptors,
    compute_sequence_aaindex_means,
    normalize_aaindex_codes,
    validate_aaindex_codes,
)

__all__ = [
    "AAIndexDescriptors",
    "aaindex_mean_descriptors",
    "compute_sequence_aaindex_means",
    "normalize_aaindex_codes",
    "validate_aaindex_codes",
]
