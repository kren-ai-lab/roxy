
from __future__ import annotations

from typing import Dict

from .base import BaseDescriptorEngine
from .sequences import GlobalSequenceDescriptors
from .structures import BasicStructureDescriptors
from .compounds import BasicMoleculeDescriptors

DESCRIPTOR_REGISTRY: Dict[str, BaseDescriptorEngine] = {
    "seq_global": GlobalSequenceDescriptors(),
    "struct_basic": BasicStructureDescriptors(),
    "mol_basic": BasicMoleculeDescriptors(),
}
