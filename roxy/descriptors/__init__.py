"""Roxy descriptors subpackage."""

from __future__ import annotations

from . import aaindex as aaindex
from . import autocorrelation as autocorrelation
from . import complexity as complexity
from . import composition as composition
from . import ctd as ctd
from . import distribution as distribution
from . import functional_residue as functional_residue
from . import hybrid as hybrid
from . import local_repetition as local_repetition
from . import motif as motif
from . import physicochemical as physicochemical
from . import positional as positional
from . import pseudo as pseudo
from . import qso as qso
from . import run_blockiness as run_blockiness
from . import sequence_order as sequence_order
from . import sliding_window as sliding_window
from . import spacing as spacing
from . import terminal as terminal
from . import user_regex as user_regex
from .base import BaseDescriptor
from .registry import DESCRIPTOR_REGISTRY, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "register",
]
