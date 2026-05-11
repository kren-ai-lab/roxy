"""Roxy descriptors subpackage."""

from __future__ import annotations

from . import aaindex as aaindex
from . import autocorrelation as autocorrelation
from . import complexity as complexity
from . import composition as composition
from . import ctd as ctd
from . import hybrid as hybrid
from . import motif as motif
from . import physicochemical as physicochemical
from . import positional as positional
from . import pseudo as pseudo
from .aaindex import AAIndexDescriptor
from .autocorrelation import AutocorrelationDescriptor
from .base import BaseDescriptor
from .complexity import (
    EntropyComplexityDescriptor,
    LocalRepetitionDescriptor,
    RunBlockinessDescriptor,
)
from .composition import (
    AACDescriptor,
    CompositionalBiasDescriptor,
    DPCDescriptor,
    GroupedDescriptor,
    KmerFullAlphabetDescriptor,
    ReducedKmerDescriptor,
)
from .ctd import CTDClassicDescriptor, DistributionDescriptor
from .hybrid import FamilySummaryDescriptor
from .motif import (
    FunctionalResidueContentDescriptor,
    PatternDescriptor,
    SpacingDescriptor,
    UserRegexDescriptor,
)
from .physicochemical import (
    ChargeDescriptor,
    GlobalBasicDescriptor,
    HydrophobicityDescriptor,
    OrderDisorderDescriptor,
    StructuralPropensityDescriptor,
)
from .positional import (
    NormalizedPositionDescriptor,
    SlidingWindowDescriptor,
    TerminalDescriptor,
)
from .pseudo import PseAACDescriptor, QSODescriptor, SequenceOrderDescriptor
from .registry import DESCRIPTOR_REGISTRY, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "AACDescriptor",
    "AAIndexDescriptor",
    "AutocorrelationDescriptor",
    "BaseDescriptor",
    "CTDClassicDescriptor",
    "ChargeDescriptor",
    "CompositionalBiasDescriptor",
    "DPCDescriptor",
    "DistributionDescriptor",
    "EntropyComplexityDescriptor",
    "FamilySummaryDescriptor",
    "FunctionalResidueContentDescriptor",
    "GlobalBasicDescriptor",
    "GroupedDescriptor",
    "HydrophobicityDescriptor",
    "KmerFullAlphabetDescriptor",
    "LocalRepetitionDescriptor",
    "NormalizedPositionDescriptor",
    "OrderDisorderDescriptor",
    "PatternDescriptor",
    "PseAACDescriptor",
    "QSODescriptor",
    "ReducedKmerDescriptor",
    "RunBlockinessDescriptor",
    "SequenceOrderDescriptor",
    "SlidingWindowDescriptor",
    "SpacingDescriptor",
    "StructuralPropensityDescriptor",
    "TerminalDescriptor",
    "UserRegexDescriptor",
    "register",
]
