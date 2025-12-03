
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseDescriptorEngine


@dataclass
class BasicStructureDescriptors(BaseDescriptorEngine):
    """Compute basic structural descriptors from protein structures (placeholder)."""

    name: str = "basic_structure_descriptors"

    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "pdb_path" not in samples.columns:
            raise ValueError("Expected a 'pdb_path' column in samples.")

        rows = []
        for _idx, _row in samples.iterrows():
            rows.append(
                {
                    "struct_rg": float("nan"),
                    "struct_contact_density": float("nan"),
                    "struct_alpha_frac": float("nan"),
                    "struct_beta_frac": float("nan"),
                }
            )
        return pd.DataFrame(rows, index=samples.index)
