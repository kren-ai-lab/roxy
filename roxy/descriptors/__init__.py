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
from .base import BaseDescriptor
from .registry import DESCRIPTOR_REGISTRY, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "register",
]
