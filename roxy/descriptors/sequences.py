
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .base import BaseDescriptorEngine

AA_ALPHABET: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class GlobalSequenceDescriptors(BaseDescriptorEngine):
    """Compute simple global descriptors from amino acid sequences."""

    name: str = "global_sequence_descriptors"

    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "sequence" not in samples.columns:
            raise ValueError("Expected a 'sequence' column in samples.")

        seqs = samples["sequence"].astype(str)
        data: dict = {"length": seqs.str.len()}

        for aa in AA_ALPHABET:
            data[f"aa_frac_{aa}"] = seqs.apply(
                lambda s, aa=aa: s.count(aa) / len(s) if s else np.nan
            )

        return pd.DataFrame(data, index=samples.index)
