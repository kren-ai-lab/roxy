from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

class BaseDescriptorEngine(ABC):
    """
    Base class for descriptor calculators operating on a samples table.

    All descriptor engines must implement the `compute` method, which
    receives a samples DataFrame and returns a feature table (DataFrame)
    with one row per sample.
    """

    name: str

    @abstractmethod
    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptors for the given samples table.

        Parameters
        ----------
        samples :
            DataFrame with at least the columns required by the engine
            (e.g. 'sequence', 'pdb_path' or 'smiles').

        Returns
        -------
        descriptors :
            Feature table indexed like `samples`.
        """
        raise NotImplementedError
