"""K-mer descriptor family exports."""

from roxy.sequence.descriptors.kmers.dpc import (
    DipeptideDescriptors,
    dpc_descriptors,
)
from roxy.sequence.descriptors.kmers.full import (
    KMerDescriptors,
    kmer_descriptors,
)

__all__ = [
    "DipeptideDescriptors",
    "KMerDescriptors",
    "dpc_descriptors",
    "kmer_descriptors",
]
