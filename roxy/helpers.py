"""High-level convenience helpers for Roxy.

This module provides opinionated, end-to-end helpers that combine
multiple Roxy components (descriptors, EDA, projection, reporting) into
single calls.

The goal is to lower the barrier for typical workflows, while keeping
the underlying primitives (RoxyDataset, descriptor engines, EDA, viz)
fully accessible for advanced use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from roxy.core.dataset import RoxyDataset
from roxy.descriptors import (
    GlobalSequenceDescriptors,
    DESCRIPTOR_REGISTRY,
    compute_descriptors,
)
from roxy.eda.summary import build_report
from roxy.projection import project
from roxy.report import (
    dataset_report_to_markdown,
    dataset_report_to_html,
)


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


@dataclass
class SequenceDescribeResult:
    """Container for the outputs of a sequence description workflow.

    Attributes
    ----------
    dataset :
        The RoxyDataset instance used in the workflow, including any
        attached feature blocks.
    feature_keys :
        Names of the feature blocks that were used to assemble ``X``.
    X :
        Combined feature matrix used for EDA and projection.
    y :
        Target vector, if provided.
    report :
        Dataset-level report object produced by :func:`build_report`.
    markdown :
        Markdown rendering of the dataset report.
    html :
        HTML rendering of the dataset report.
    embedding :
        Optional low-dimensional embedding (e.g. PCA, UMAP, t-SNE) of
        the feature matrix.
    extra :
        Optional dictionary for any additional artefacts the helper
        might include in the future (e.g. figures, diagnostics).
    """

    dataset: RoxyDataset
    feature_keys: List[str]
    X: pd.DataFrame
    y: Optional[Union[pd.Series, np.ndarray]]
    report: Any
    markdown: str
    html: str
    embedding: Optional[np.ndarray] = None
    extra: Dict[str, Any] = None  # type: ignore[assignment]


class RoxyHelpers:
    """Namespace for opinionated, high-level helper workflows.

    All methods are defined as ``@staticmethod`` so that they can be
    used either via the class itself or imported individually.

    Example
    -------
    >>> from roxy.helpers import RoxyHelpers
    >>> result = RoxyHelpers.describe_sequences(df, seq_col="sequence",
    ...                                         y="label",
    ...                                         dataset_name="toy_sequences")
    >>> result.X.shape
    >>> result.markdown[:500]
    """

    # ------------------------------------------------------------------
    # Sequence-centric helper
    # ------------------------------------------------------------------

    @staticmethod
    def describe_sequences(
        df: pd.DataFrame,
        *,
        seq_col: str = "sequence",
        y: Optional[Union[str, pd.Series]] = None,
        dataset_name: Optional[str] = None,
        use_aaindex: bool = True,
        feature_key_prefix: str = "",
        task_type: Optional[str] = None,
        project_method: Optional[str] = "pca",
        n_components: int = 2,
        random_state: int = 42,
    ) -> SequenceDescribeResult:
        """End-to-end helper for describing protein sequences.

        This helper performs the following steps:

        1. Builds a :class:`RoxyDataset` from the input DataFrame.
        2. Computes global sequence descriptors via
           :class:`GlobalSequenceDescriptors`.
        3. Optionally computes AAIndex-based descriptors if a
           ``"seq_aaindex"`` engine is registered.
        4. Combines selected feature blocks into a single feature matrix.
        5. Builds a :class:`~roxy.core.report.DatasetReport` using EDA
           summary utilities.
        6. Renders Markdown and HTML reports.
        7. Optionally computes a low-dimensional embedding via the
           projection module (e.g. PCA).

        Parameters
        ----------
        df :
            Input DataFrame that must contain at least a sequence column
            (default name: ``"sequence"``). Additional columns are kept
            as metadata.
        seq_col :
            Name of the column in ``df`` that contains amino-acid
            sequences.
        y :
            Target labels. If a string is provided, it is interpreted as
            the name of a column in ``df``. If a Series is provided, it
            is used directly.
        dataset_name :
            Optional name for the dataset, used in reports.
        use_aaindex :
            If ``True`` and a ``"seq_aaindex"`` engine is present in
            :data:`DESCRIPTOR_REGISTRY`, AAIndex-based descriptors are
            also computed and included.
        feature_key_prefix :
            Optional prefix applied to feature block names when attaching
            them to the dataset.
        task_type :
            Task type hint (``"classification"`` or ``"regression"``).
            If ``None``, a simple heuristic is used when labels are
            available.
        project_method :
            If not ``None``, name of the projection method to pass to
            :func:`roxy.projection.project` (e.g. ``"pca"``, ``"umap"``,
            ``"tsne"``). If ``None``, no embedding is computed.
        n_components :
            Number of components for the projection method.
        random_state :
            Random seed passed to projection methods that support it.

        Returns
        -------
        SequenceDescribeResult
            Container with the dataset, combined features, report,
            textual renderings and optional embedding.

        Notes
        -----
        This helper is designed as a “happy path” for quick inspection
        of sequence datasets. For finer control, use the underlying
        components directly (RoxyDataset, descriptor engines, EDA,
        projection, viz, report).
        """
        if seq_col not in df.columns:
            raise KeyError(
                f"Sequence column {seq_col!r} not found in input DataFrame. "
                f"Available columns: {', '.join(df.columns)}"
            )

        # ---- Target handling ------------------------------------------------
        if isinstance(y, str):
            if y not in df.columns:
                raise KeyError(
                    f"Target column {y!r} not found in input DataFrame."
                )
            y_series: Optional[pd.Series] = df[y].copy()
        elif isinstance(y, pd.Series):
            y_series = y
        else:
            y_series = None

        # ---- Build dataset --------------------------------------------------
        ds_name = dataset_name or "sequences"
        dataset = RoxyDataset(samples=df.copy(), y=y_series, name=ds_name)

        # ---- Descriptor engines --------------------------------------------
        feature_keys: List[str] = []

        # Always compute global sequence descriptors
        engines: List[str] = []

        # Ensure seq_global is in the registry; fall back to direct engine
        if "seq_global" in DESCRIPTOR_REGISTRY:
            engines.append("seq_global")
        else:
            # Directly compute and attach if not registered for some reason
            seq_engine = GlobalSequenceDescriptors()
            X_seq = seq_engine.compute(dataset.samples)
            key = f"{feature_key_prefix}seq_global"
            dataset.add_features(key, X_seq)
            feature_keys.append(key)

        # Optionally compute AAIndex-based descriptors via registry
        if use_aaindex and "seq_aaindex" in DESCRIPTOR_REGISTRY:
            engines.append("seq_aaindex")

        if engines:
            compute_descriptors(
                dataset,
                engines=engines,
                feature_key_prefix=feature_key_prefix,
            )
            for name in engines:
                feature_keys.append(f"{feature_key_prefix}{name}")

        # ---- Combine features into a single matrix -------------------------
        X_all, y_aligned = dataset.to_Xy(feature_keys=feature_keys)

        # ---- Infer task type if not provided -------------------------------
        inferred_task_type = task_type
        if inferred_task_type is None and y_aligned is not None:
            inferred_task_type = RoxyHelpers._infer_task_type(y_aligned)

        # ---- Build EDA report ----------------------------------------------
        report = build_report(
            X_all,
            y=y_aligned,
            dataset_name=ds_name,
            task_type=inferred_task_type,
        )

        md = dataset_report_to_markdown(report)
        html = dataset_report_to_html(report)

        # ---- Optional projection -------------------------------------------
        embedding: Optional[np.ndarray] = None
        if project_method is not None:
            embedding = project(
                X_all,
                method=project_method,
                n_components=n_components,
                random_state=random_state,
            )

        return SequenceDescribeResult(
            dataset=dataset,
            feature_keys=feature_keys,
            X=X_all,
            y=y_aligned,
            report=report,
            markdown=md,
            html=html,
            embedding=embedding,
            extra={},
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def basic_eda_report(
        X: pd.DataFrame,
        y: Optional[ArrayLike] = None,
        *,
        dataset_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Tuple[Any, str, str]:
        """Build a basic EDA report and its textual renderings.

        This helper is agnostic to the origin of the features. It simply
        wraps :func:`build_report` and the reporting utilities.

        Parameters
        ----------
        X :
            Feature matrix as a pandas DataFrame.
        y :
            Optional target vector aligned with ``X``. If provided,
            feature–target associations are included in the report.
        dataset_name :
            Optional dataset name, used in report headers.
        task_type :
            Task type hint (``"classification"`` or ``"regression"``).
            If ``None`` and ``y`` is provided, a simple heuristic is
            used to infer the type.

        Returns
        -------
        report :
            The DatasetReport object.
        markdown :
            Markdown rendering of the report.
        html :
            HTML rendering of the report.
        """
        y_series: Optional[pd.Series]
        if y is None:
            y_series = None
        elif isinstance(y, pd.Series):
            y_series = y
        else:
            y_series = pd.Series(y, index=X.index)

        inferred_task_type = task_type
        if inferred_task_type is None and y_series is not None:
            inferred_task_type = RoxyHelpers._infer_task_type(y_series)

        report = build_report(
            X,
            y=y_series,
            dataset_name=dataset_name,
            task_type=inferred_task_type,
        )
        md = dataset_report_to_markdown(report)
        html = dataset_report_to_html(report)
        return report, md, html

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_task_type(y: Union[pd.Series, np.ndarray]) -> str:
        """Heuristic to infer task type from the target vector.

        - If the number of unique values is relatively small (e.g.
          ``<= 20`` or less than 5 % of the sample size), we treat it as
          a classification problem.
        - Otherwise, we treat it as regression.

        Parameters
        ----------
        y :
            Target vector.

        Returns
        -------
        str
            Either ``"classification"`` or ``"regression"``.
        """
        if isinstance(y, np.ndarray):
            y_arr = y
        else:
            y_arr = y.to_numpy()

        n = len(y_arr)
        unique_vals = np.unique(y_arr)
        n_unique = len(unique_vals)

        if n_unique <= 20 or n_unique <= 0.05 * n:
            return "classification"
        return "regression"
