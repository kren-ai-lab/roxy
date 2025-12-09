from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from roxy.core.exceptions import DatasetError
from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RoxyDataset:
    """
    Multimodal container for molecular data used in exploratory analysis.

    This class is the central data structure of Roxy. It stores:

    - A table of raw sample-level information (:attr:`samples`), typically
      including identifiers and raw modalities such as ``sequence``,
      ``pdb_path`` or ``smiles``.
    - An optional target variable (:attr:`y`) used for supervised tasks.
    - Optional dataset-level metadata (:attr:`metadata`), such as task type,
      source or experimental conditions.
    - One or more feature tables (:attr:`features`) produced by descriptor
      engines or external encoders (e.g. Sylphy embeddings).

    Parameters
    ----------
    samples
        Table with one row per sample (e.g. protein, molecule, complex).
        The index is expected to uniquely identify samples and will be used
        to align feature tables.
    y
        Optional target variable. Must be a pandas Series or a 1D numpy
        array whose length matches the number of rows in ``samples``. If a
        Series is provided, it will be aligned by index.
    metadata
        Optional dataset-level metadata. This can be any dictionary, but is
        typically used to store fields such as ``task_type``, ``source``, etc.
    name
        Optional dataset name, used mainly for reporting and logging.
    features
        Optional mapping from feature block names (e.g. ``"seq_desc"``,
        ``"struct_desc"``, ``"sylphy_embed"``) to feature tables. Feature
        tables must be pandas DataFrames indexed like ``samples``.
    """

    samples: pd.DataFrame
    y: Optional[pd.Series | np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return int(self.samples.shape[0])

    @property
    def n_raw_columns(self) -> int:
        """Number of raw columns in the samples table."""
        return int(self.samples.shape[1])

    @property
    def feature_blocks(self) -> List[str]:
        """Names of registered feature blocks."""
        return list(self.features.keys())

    # -------------------------------------------------------------------------
    # Construction & validation
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate internal consistency after initialisation."""
        self._validate_samples()
        self._normalize_y()
        self._validate_features()
        logger.info(
            "Initialised RoxyDataset(name=%r, n_samples=%d, n_raw_columns=%d, "
            "feature_blocks=%d)",
            self.name,
            self.n_samples,
            self.n_raw_columns,
            len(self.features),
        )

    def _validate_samples(self) -> None:
        """Validate the main samples table.

        Ensures that ``samples`` is a DataFrame with a unique index, which
        is required to align feature blocks and targets in a predictable way.
        """
        if not isinstance(self.samples, pd.DataFrame):
            msg = "`samples` must be a pandas DataFrame."
            logger.error("%s Got %r", msg, type(self.samples))
            raise DatasetError(msg)

        if self.samples.index.has_duplicates:
            msg = (
                "The index of `samples` contains duplicates. RoxyDataset "
                "expects a unique index to align feature tables."
            )
            logger.error(msg)
            raise DatasetError(msg)

    def _normalize_y(self) -> None:
        """Normalise and align the target vector ``y`` with ``samples``."""
        if self.y is None:
            return

        if isinstance(self.y, pd.Series):
            # Align by index and ensure length consistency
            before_len = len(self.y)
            self.y = self.y.reindex(self.samples.index)
            after_len = len(self.y)
            if before_len != after_len:
                logger.warning(
                    "Target Series was reindexed from %d to %d entries to match samples.",
                    before_len,
                    after_len,
                )
        else:
            self.y = np.asarray(self.y)
            if self.y.ndim != 1:
                msg = "`y` must be 1-dimensional."
                logger.error(msg)
                raise DatasetError(msg)
            if self.y.shape[0] != self.n_samples:
                msg = (
                    f"Length of `y` ({self.y.shape[0]}) does not match number "
                    f"of samples ({self.n_samples})."
                )
                logger.error(msg)
                raise DatasetError(msg)

    def _validate_features(self) -> None:
        """Validate all registered feature blocks."""
        for key, X in self.features.items():
            if not isinstance(X, pd.DataFrame):
                msg = (
                    f"Feature block '{key}' must be a pandas DataFrame, "
                    f"got {type(X)!r}."
                )
                logger.error(msg)
                raise DatasetError(msg)

            # Require that indices are a subset of the samples index
            if not X.index.isin(self.samples.index).all():
                msg = (
                    f"Some indices in feature block '{key}' are not present "
                    "in `samples`."
                )
                logger.error(msg)
                raise DatasetError(msg)

    # -------------------------------------------------------------------------
    # Modality helpers
    # -------------------------------------------------------------------------

    def has_sequence(self, column: str = "sequence") -> bool:
        """Return True if the samples table contains a sequence column."""
        return column in self.samples.columns

    def has_structure(self, column: str = "pdb_path") -> bool:
        """Return True if the samples table contains a structure column."""
        return column in self.samples.columns

    def has_compound(self, column: str = "smiles") -> bool:
        """Return True if the samples table contains a compound column."""
        return column in self.samples.columns

    # -------------------------------------------------------------------------
    # Feature management
    # -------------------------------------------------------------------------

    def add_features(
        self,
        key: str,
        X: pd.DataFrame,
        *,
        validate_index: bool = True,
        overwrite: bool = True,
    ) -> None:
        """
        Register a feature table under the given key.

        Parameters
        ----------
        key
            Name of the feature block (e.g. "seq_desc", "struct_basic",
            "sylphy_embed").
        X
            Feature table with one row per sample. Its index must be
            compatible with ``samples.index``.
        validate_index
            If True, ensure that all indices in ``X`` exist in ``samples``.
        overwrite
            If False and the key already exists, raise a DatasetError instead
            of overwriting the existing feature block.
        """
        if not isinstance(X, pd.DataFrame):
            msg = "`X` must be a pandas DataFrame."
            logger.error("%s Got %r", msg, type(X))
            raise DatasetError(msg)

        if validate_index:
            if not X.index.isin(self.samples.index).all():
                msg = (
                    f"Some indices in feature block '{key}' are not present "
                    "in `samples`."
                )
                logger.error(msg)
                raise DatasetError(msg)

        if not overwrite and key in self.features:
            msg = (
                f"Feature block '{key}' already exists and `overwrite` is False."
            )
            logger.error(msg)
            raise DatasetError(msg)

        # Align to samples index (inner join semantics)
        X_aligned = X.reindex(self.samples.index)
        self.features[key] = X_aligned
        logger.info(
            "Registered feature block '%s' with shape %s.",
            key,
            X_aligned.shape,
        )

    def get_feature_table(
        self,
        keys: Optional[Iterable[str]] = None,
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Return a merged feature table for the specified feature blocks.

        Parameters
        ----------
        keys
            Iterable of feature block names. If None, all registered blocks
            are used.
        how
            Merge strategy for aligning feature blocks:
            - "inner": keep only samples present in all blocks.
            - "left":  start from ``samples.index`` and align feature tables
                       to it, potentially introducing missing values.

        Returns
        -------
        X
            Merged feature DataFrame.

        Raises
        ------
        DatasetError
            If no feature tables are registered, requested blocks are
            missing, or an unsupported merge strategy is used.
        """
        if not self.features:
            msg = "No feature tables have been registered."
            logger.error(msg)
            raise DatasetError(msg)

        if keys is None:
            keys = self.feature_blocks
        else:
            keys = list(keys)
            missing = [k for k in keys if k not in self.features]
            if missing:
                msg = (
                    f"Feature blocks not found: {', '.join(missing)}. "
                    f"Available blocks: {', '.join(self.feature_blocks)}"
                )
                logger.error(msg)
                raise DatasetError(msg)

        dfs: List[pd.DataFrame] = [self.features[k] for k in keys]

        if how == "inner":
            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.join(df, how="inner")
        elif how == "left":
            merged = pd.DataFrame(index=self.samples.index)
            for df in dfs:
                merged = merged.join(df, how="left")
        else:
            msg = f"Unsupported merge strategy: {how!r}"
            logger.error(msg)
            raise DatasetError(msg)

        return merged

    def to_Xy(
        self,
        feature_keys: Optional[Iterable[str]] = None,
        *,
        how: str = "inner",
        dropna: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series | np.ndarray]]:
        """
        Return a feature matrix X and target y aligned on the same samples.

        Parameters
        ----------
        feature_keys
            Feature blocks to include. If None, all registered blocks are used.
        how
            Merge strategy passed to :meth:`get_feature_table`.
        dropna
            If True, drop rows with any missing values in X or y.

        Returns
        -------
        X
            Merged feature table.
        y
            Target variable aligned to X's index, or None if no target
            was provided.

        Raises
        ------
        DatasetError
            If an internal inconsistency is detected between the length
            of ``y`` and the number of samples.
        """
        X = self.get_feature_table(keys=feature_keys, how=how)

        if self.y is None:
            return X, None

        # Ensure y is a Series aligned to samples index
        if isinstance(self.y, pd.Series):
            y_aligned = self.y.reindex(self.samples.index)
        else:
            if self.y.shape[0] != self.n_samples:
                msg = (
                    "Internal inconsistency: length of `y` does not "
                    "match samples."
                )
                logger.error(msg)
                raise DatasetError(msg)
            y_aligned = pd.Series(self.y, index=self.samples.index)

        # Reindex y to match the merged feature table
        y_aligned = y_aligned.reindex(X.index)

        if dropna:
            mask = ~X.isna().any(axis=1)
            mask &= ~y_aligned.isna()
            X = X[mask]
            y_aligned = y_aligned[mask]

        return X, y_aligned

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def copy(self, *, deep: bool = True) -> "RoxyDataset":
        """
        Return a (deep) copy of the dataset.

        Parameters
        ----------
        deep
            If True, copy samples, features and metadata. If False, only
            the top-level structure is copied (shallow copy).
        """
        if not deep:
            return replace(self)

        samples = self.samples.copy(deep=True)
        y = self.y.copy() if isinstance(self.y, pd.Series) else (
            None if self.y is None else np.copy(self.y)
        )
        metadata = dict(self.metadata)
        features = {k: v.copy(deep=True) for k, v in self.features.items()}

        return RoxyDataset(
            samples=samples,
            y=y,
            metadata=metadata,
            name=self.name,
            features=features,
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        info = [
            f"RoxyDataset(name={self.name!r}, n_samples={self.n_samples}, "
            f"n_raw_columns={self.n_raw_columns})",
            f"  feature_blocks: {', '.join(self.feature_blocks) if self.features else 'none'}",
            f"  has_y: {self.y is not None}",
        ]
        return "<" + "\n".join(info) + ">"
