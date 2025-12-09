"""Optional interactive dashboard helpers based on Plotly.

This module provides a light abstraction around Plotly figures for
interactive exploration of feature matrices and embeddings.

The intent is *not* to prescribe a fixed web framework (Dash, Panel,
Streamlit, etc.) but to provide reusable figures and a small
``RoxyDashboard`` helper that other frontends can embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)

try:  # Optional dependency
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = None  # type: ignore
    go = None  # type: ignore


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _require_plotly() -> None:
    """Raise an informative error if Plotly is not available."""
    if px is None or go is None:
        msg = (
            "Plotly is required for interactive dashboard helpers. "
            "Install it with `pip install plotly` or `pip install roxy[viz]`."
        )
        logger.error("_require_plotly: %s", msg)
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Standalone figure helpers
# ---------------------------------------------------------------------------


def interactive_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    y: Optional[pd.Series] = None,
    *,
    max_classes: int = 15,
    nbins: int = 30,
    title: Optional[str] = None,
) -> "go.Figure":
    """Create an interactive histogram for a feature.

    Parameters
    ----------
    df :
        Input DataFrame containing the feature.
    feature :
        Column name in ``df`` to visualise.
    y :
        Optional label Series with the same index as ``df``. If provided,
        class-conditional histograms are drawn.
    max_classes :
        Maximum number of classes to display separately if ``y`` is given.
    nbins :
        Number of histogram bins.
    title :
        Optional figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured Plotly figure.
    """
    _require_plotly()

    if feature not in df.columns:
        msg = f"Feature {feature!r} not found in df columns."
        logger.error("interactive_feature_distribution: %s", msg)
        raise KeyError(msg)

    data = df[[feature]].copy()
    if y is not None:
        if len(y) != len(df):
            msg = "Length of y must match number of rows in df."
            logger.error("interactive_feature_distribution: %s", msg)
            raise ValueError(msg)

        data["label"] = y.values

        labels = data["label"].unique()
        if len(labels) > max_classes:
            logger.info(
                "interactive_feature_distribution: truncating labels from %d to %d.",
                len(labels),
                max_classes,
            )
            labels = labels[:max_classes]

        fig = px.histogram(
            data.query("label in @labels"),
            x=feature,
            color="label",
            nbins=nbins,
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            title=title or f"Distribution of {feature} by label",
        )
    else:
        fig = px.histogram(
            data,
            x=feature,
            nbins=nbins,
            opacity=0.8,
            marginal="box",
            title=title or f"Distribution of {feature}",
        )

    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
    )
    return fig


def interactive_embedding(
    embedding: ArrayLike,
    y: Optional[pd.Series] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    method: str = "embedding",
    hover_data: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
) -> "go.Figure":
    """Create an interactive 2D scatter plot of an embedding.

    Parameters
    ----------
    embedding :
        2D coordinates of shape (n_samples, 2). Can be a NumPy array or
        a DataFrame.
    y :
        Optional label Series for colour-coding points.
    labels :
        Optional names for embedding dimensions (x, y).
    method :
        Name of the embedding method, used in axis labels and title
        (e.g. ``"PCA"``, ``"UMAP"``, ``"t-SNE"``).
    hover_data :
        Optional DataFrame with additional columns to show on hover.
        Must have the same index length as the embedding.
    title :
        Optional figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured Plotly scatter plot.
    """
    _require_plotly()

    arr = embedding.to_numpy() if isinstance(embedding, pd.DataFrame) else np.asarray(embedding)
    if arr.ndim != 2 or arr.shape[1] != 2:
        msg = f"Expected embedding with shape (n_samples, 2), got {arr.shape!r}."
        logger.error("interactive_embedding: %s", msg)
        raise ValueError(msg)

    n = arr.shape[0]
    df = pd.DataFrame({"dim1": arr[:, 0], "dim2": arr[:, 1]})

    if labels is not None and len(labels) == 2:
        x_label, y_label = labels
    else:
        x_label, y_label = f"{method} 1", f"{method} 2"

    if y is not None:
        if len(y) != n:
            msg = "Length of y must match number of rows in embedding."
            logger.error("interactive_embedding: %s", msg)
            raise ValueError(msg)
        df["label"] = y.values

    if hover_data is not None:
        if len(hover_data) != n:
            msg = "hover_data must have the same number of rows as embedding."
            logger.error("interactive_embedding: %s", msg)
            raise ValueError(msg)
        for col in hover_data.columns:
            df[col] = hover_data[col].values

    logger.debug(
        "interactive_embedding: plotting %d points (labels=%s, hover_cols=%d).",
        n,
        "yes" if "label" in df.columns else "no",
        len(df.columns) - 2,
    )

    fig = px.scatter(
        df,
        x="dim1",
        y="dim2",
        color="label" if "label" in df.columns else None,
        hover_data=[c for c in df.columns if c not in {"dim1", "dim2"}],
        title=title or f"{method} embedding",
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


# ---------------------------------------------------------------------------
# Minimal dashboard container
# ---------------------------------------------------------------------------


@dataclass
class RoxyDashboard:
    """Lightweight container for interactive visualisations.

    This class organises a feature matrix, optional labels and one or
    more precomputed embeddings. It exposes convenience methods to
    generate Plotly figures that can be embedded in a web application.

    Parameters
    ----------
    X :
        Feature matrix as a pandas DataFrame.
    y :
        Optional label vector aligned with ``X``.
    embeddings :
        Optional mapping from embedding name (e.g. ``"pca"``, ``"umap"``)
        to 2D NumPy arrays or DataFrames with shape (n_samples, 2).
    """

    X: pd.DataFrame
    y: Optional[pd.Series] = None
    embeddings: Dict[str, ArrayLike] = field(default_factory=dict)

    def add_embedding(self, name: str, embedding: ArrayLike) -> None:
        """Register an embedding under a given name."""
        logger.debug("RoxyDashboard.add_embedding: registering %r.", name)
        self.embeddings[name] = embedding

    # ---- Figure factories -------------------------------------------------

    def figure_distribution(
        self,
        feature: str,
        *,
        nbins: int = 30,
        max_classes: int = 15,
        title: Optional[str] = None,
    ) -> "go.Figure":
        """Interactive histogram for a feature, with optional stratification."""
        logger.info(
            "RoxyDashboard.figure_distribution: feature=%r, nbins=%d.",
            feature,
            nbins,
        )
        return interactive_feature_distribution(
            self.X,
            feature=feature,
            y=self.y,
            nbins=nbins,
            max_classes=max_classes,
            title=title,
        )

    def figure_embedding(
        self,
        name: str,
        *,
        labels: Optional[Sequence[str]] = None,
        method_label: Optional[str] = None,
        hover_columns: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
    ) -> "go.Figure":
        """Interactive 2D scatter for a named embedding.

        Parameters
        ----------
        name :
            Name of the embedding as registered via :meth:`add_embedding`.
        labels :
            Optional names for the embedding dimensions.
        method_label :
            Human-readable name for the method (e.g. ``"PCA"``). If not
            provided, the embedding name is used.
        hover_columns :
            Optional iterable of column names in ``X`` to include as
            hover metadata.
        title :
            Optional figure title.
        """
        if name not in self.embeddings:
            msg = (
                f"Embedding {name!r} not found. "
                f"Available embeddings: {', '.join(self.embeddings)}"
            )
            logger.error("RoxyDashboard.figure_embedding: %s", msg)
            raise KeyError(msg)

        emb = self.embeddings[name]
        method = method_label or name

        hover_df: Optional[pd.DataFrame] = None
        if hover_columns is not None:
            cols = list(hover_columns)
            logger.debug(
                "RoxyDashboard.figure_embedding: using hover columns %s.", cols
            )
            hover_df = self.X[cols].copy()

        logger.info(
            "RoxyDashboard.figure_embedding: embedding=%r, method_label=%r.",
            name,
            method,
        )
        return interactive_embedding(
            emb,
            y=self.y,
            labels=labels,
            method=method,
            hover_data=hover_df,
            title=title,
        )
