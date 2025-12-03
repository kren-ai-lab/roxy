
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseDescriptorEngine


@dataclass
class BasicMoleculeDescriptors(BaseDescriptorEngine):
    """Compute basic molecular descriptors from SMILES strings (placeholder)."""

    name: str = "basic_molecule_descriptors"

    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "smiles" not in samples.columns:
            raise ValueError("Expected a 'smiles' column in samples.")

        rows = []
        for _idx, _row in samples.iterrows():
            rows.append(
                {
                    "mol_weight": float("nan"),
                    "mol_logp": float("nan"),
                    "mol_tpsa": float("nan"),
                }
            )
        return pd.DataFrame(rows, index=samples.index)
