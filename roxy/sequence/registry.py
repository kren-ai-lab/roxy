"""Registry definitions for the sequence-only descriptor architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from roxy.sequence.descriptors._base import SequenceFeatureMap
from roxy.sequence.descriptors.aaindex.aaindex import aaindex_mean_descriptors
from roxy.sequence.descriptors.basic.composition import aac_descriptors
from roxy.sequence.descriptors.basic.global_basic import global_basic_descriptors
from roxy.sequence.descriptors.basic.grouped import grouped_composition_descriptors
from roxy.sequence.descriptors.kmers.dpc import dpc_descriptors

SequenceDescriptorCallable = Callable[..., SequenceFeatureMap]


@dataclass(frozen=True)
class SequenceDescriptorRegistration:
    """Descriptor-family registration entry for the sequence registry."""

    name: str
    handler: SequenceDescriptorCallable
    description: str


SEQUENCE_DESCRIPTOR_BLOCK_REGISTRY: Dict[str, SequenceDescriptorRegistration] = {
    "global_basic": SequenceDescriptorRegistration(
        name="global_basic",
        handler=global_basic_descriptors,
        description="Global/basic scalar sequence descriptors.",
    ),
    "aac": SequenceDescriptorRegistration(
        name="aac",
        handler=aac_descriptors,
        description="Amino-acid composition descriptors.",
    ),
    "grouped": SequenceDescriptorRegistration(
        name="grouped",
        handler=grouped_composition_descriptors,
        description="Grouped residue composition fractions and ratios.",
    ),
    "aaindex_mean": SequenceDescriptorRegistration(
        name="aaindex_mean",
        handler=aaindex_mean_descriptors,
        description="Per-index AAIndex mean descriptors.",
    ),
    "dpc": SequenceDescriptorRegistration(
        name="dpc",
        handler=dpc_descriptors,
        description="Dipeptide composition descriptors.",
    ),
}


def list_available_descriptors() -> List[str]:
    """Return the stable names of registered sequence descriptor blocks."""
    return sorted(SEQUENCE_DESCRIPTOR_BLOCK_REGISTRY.keys())


def get_descriptor_registration(name: str) -> SequenceDescriptorRegistration:
    """Return a registered sequence descriptor-family entry by name."""
    return SEQUENCE_DESCRIPTOR_BLOCK_REGISTRY[name]


def get_descriptor_handler(name: str) -> SequenceDescriptorCallable:
    """Return the callable that implements a registered descriptor block."""
    return get_descriptor_registration(name).handler


def describe_with_block(name: str, sequence: str, **kwargs: object) -> SequenceFeatureMap:
    """Execute a registered descriptor block for a single sequence."""
    handler = get_descriptor_handler(name)
    return handler(sequence, **kwargs)


__all__ = [
    "SEQUENCE_DESCRIPTOR_BLOCK_REGISTRY",
    "SequenceDescriptorRegistration",
    "describe_with_block",
    "get_descriptor_handler",
    "get_descriptor_registration",
    "list_available_descriptors",
]
