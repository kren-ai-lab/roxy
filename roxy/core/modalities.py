
from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Supported data modalities for Roxy."""

    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    COMPOUND = "compound"
