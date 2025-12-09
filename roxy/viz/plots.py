"""Matplotlib-based plotting utilities for Roxy.

This module focuses on lightweight, composable plotting helpers that
operate on pandas DataFrames and NumPy arrays. The goal is to provide
standard visualisations that can be reused across notebooks, scripts,
and higher-level dashboards.

Currently supported plot types include:

- univariate feature distributions (histograms),
- boxplots of features stratified by label,
- 2D scatter plots in feature space,
- 2D scatter plots of low-dimensional embeddings (e.g. PCA, UMAP, t-SNE),
- correlation heatmaps for numeric features.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _ensure_1d(a: ArrayLike) -> np.ndarray:
    """Convert input to a 1D NumPy array."""
    if isinstance(a, (pd.Series, pd.Index)):
        return a.to_numpy()
    arr = np.asarray(a)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.ravel()


def _ensure_2d(a: ArrayLike) -> np.ndarray:
    """Convert input to a 2D NumPy array."""
    if isinstance(a, pd.DataFrame):
        return a.to_numpy()
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _get_fig_ax(ax: Optional[Axes] = None) -> tuple[Figure, Axes]:
    """Return a (figure, axes) pair, creating them if needed."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


# ---------------------------------------------------------------------------
# Univariate distributions
# ---------------------------------------------------------------------------


def plot_feature_distribution(
    X: pd.DataFrame,
    feature: str,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    *,
    bins: int = 30,
    density: bool = False,
    max_classes: int = 10,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Plot the distribution of a single feature.

    If a label vector ``y`` is provided, class-conditional histograms are
    overlaid. This is useful to quickly assess class separation in a
    given feature.

    Parameters
    ----------
    X :
        Feature matrix as a pandas DataFrame.
    feature :
        Column name in ``X`` to plot.
    y :
        Optional labels for the samples. Can be a pandas Series or
        NumPy array. If provided, class-conditional histograms are drawn.
    bins :
        Number of histogram bins.
    density :
        If ``True``, normalise counts to form a probability density.
    max_classes :
        Maximum number of distinct label values to show separately. If
        more than this limit, only the first ``max_classes`` (by sorted
        unique label) are displayed.
    ax :
        Optional Matplotlib Axes object. If ``None``, a new figure and axes
        are created.
    title :
        Optional plot title. If omitted, a default title is generated.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if feature not in X.columns:
        msg = f"Feature {feature!r} not found in X columns."
        logger.error("plot_feature_distribution: %s", msg)
        raise KeyError(msg)

    fig, ax = _get_fig_ax(ax)
    values = X[feature].dropna()

    if y is None:
        logger.debug(
            "plot_feature_distribution: plotting feature %r without labels.", feature
        )
        ax.hist(values, bins=bins, density=density, alpha=0.8)
        ax.set_ylabel("Density" if density else "Count")
        if title is None:
            title = f"Distribution of {feature}"
    else:
        y_arr = _ensure_1d(y)
        if len(y_arr) != len(X):
            msg = "Length of y must match number of rows in X."
            logger.error("plot_feature_distribution: %s", msg)
            raise ValueError(msg)

        # Align y with non-missing values of the feature
        mask = X[feature].notna().to_numpy()
        values = values.to_numpy()
        y_masked = y_arr[mask]

        unique_labels = np.unique(y_masked)
        if len(unique_labels) > max_classes:
            logger.info(
                "plot_feature_distribution: truncating labels from %d to %d.",
                len(unique_labels),
                max_classes,
            )
            unique_labels = unique_labels[:max_classes]

        logger.debug(
            "plot_feature_distribution: plotting feature %r for %d classes.",
            feature,
            len(unique_labels),
        )
        for lab in unique_labels:
            lab_mask = y_masked == lab
            ax.hist(
                values[lab_mask],
                bins=bins,
                density=density,
                alpha=0.7,
                label=str(lab),
            )

        ax.set_ylabel("Density" if density else "Count")
        ax.legend(title="label")
        if title is None:
            title = f"Distribution of {feature} by label"

    ax.set_xlabel(feature)
    if title is not None:
        ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Boxplot by label
# ---------------------------------------------------------------------------


def plot_feature_boxplot(
    X: pd.DataFrame,
    feature: str,
    y: Union[pd.Series, np.ndarray],
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    rotate_xticks: bool = True,
) -> Axes:
    """Plot a boxplot of a feature stratified by label.

    Parameters
    ----------
    X :
        Feature matrix as a pandas DataFrame.
    feature :
        Column name in ``X`` to plot.
    y :
        Labels for each sample, used to define boxplot groups.
    ax :
        Optional Matplotlib Axes object. If ``None``, a new figure and axes
        are created.
    title :
        Optional plot title. If omitted, a default title is generated.
    rotate_xticks :
        If ``True``, rotate x-axis tick labels by 45 degrees for readability.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if feature not in X.columns:
        msg = f"Feature {feature!r} not found in X columns."
        logger.error("plot_feature_boxplot: %s", msg)
        raise KeyError(msg)

    y_arr = _ensure_1d(y)
    if len(y_arr) != len(X):
        msg = "Length of y must match number of rows in X."
        logger.error("plot_feature_boxplot: %s", msg)
        raise ValueError(msg)

    fig, ax = _get_fig_ax(ax)

    data = pd.DataFrame({"feature": X[feature], "label": y_arr}).dropna()
    groups = [grp["feature"].values for _, grp in data.groupby("label")]
    labels = [str(lbl) for lbl in data["label"].unique()]

    logger.debug(
        "plot_feature_boxplot: plotting feature %r for %d groups.",
        feature,
        len(labels),
    )

    ax.boxplot(groups, labels=labels)
    ax.set_xlabel("Label")
    ax.set_ylabel(feature)
    if rotate_xticks:
        ax.set_xticklabels(labels, rotation=45, ha="right")
    if title is None:
        title = f"{feature} by label"
    ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def plot_scatter_features(
    X: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Scatter plot of two features, optionally coloured by label.

    Parameters
    ----------
    X :
        Feature matrix as a pandas DataFrame.
    x_feature, y_feature :
        Names of the columns in ``X`` to plot on the x and y axes.
    y :
        Optional labels for colour-coding points.
    ax :
        Optional Matplotlib Axes object. If ``None``, a new figure and axes
        are created.
    title :
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    for feat in (x_feature, y_feature):
        if feat not in X.columns:
            msg = f"Feature {feat!r} not found in X columns."
            logger.error("plot_scatter_features: %s", msg)
            raise KeyError(msg)

    fig, ax = _get_fig_ax(ax)

    x_vals = X[x_feature].to_numpy()
    y_vals = X[y_feature].to_numpy()

    if y is None:
        logger.debug(
            "plot_scatter_features: plotting %r vs %r without labels.",
            x_feature,
            y_feature,
        )
        ax.scatter(x_vals, y_vals, alpha=0.8)
    else:
        y_arr = _ensure_1d(y)
        if len(y_arr) != len(X):
            msg = "Length of y must match number of rows in X."
            logger.error("plot_scatter_features: %s", msg)
            raise ValueError(msg)

        unique_labels = np.unique(y_arr)
        logger.debug(
            "plot_scatter_features: plotting %r vs %r for %d classes.",
            x_feature,
            y_feature,
            len(unique_labels),
        )
        for lab in unique_labels:
            mask = y_arr == lab
            ax.scatter(
                x_vals[mask],
                y_vals[mask],
                alpha=0.8,
                label=str(lab),
            )
        ax.legend(title="label")

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    if title is None:
        title = f"{y_feature} vs {x_feature}"
    ax.set_title(title)

    return ax


def plot_embedding(
    embedding: ArrayLike,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
) -> Axes:
    """Scatter plot of a 2D embedding (e.g. PCA, UMAP, t-SNE).

    Parameters
    ----------
    embedding :
        2D coordinates of shape (n_samples, 2). Can be a NumPy array or
        a pandas DataFrame with two columns.
    y :
        Optional labels for colour-coding points.
    ax :
        Optional Matplotlib Axes object. If ``None``, a new figure and axes
        are created.
    title :
        Optional plot title. If omitted, a generic title is used.
    labels :
        Optional names for the two embedding dimensions, used for axis
        labels. If ``None``, defaults to ``["dim1", "dim2"]``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    emb = _ensure_2d(embedding)
    if emb.shape[1] != 2:
        msg = f"Expected embedding with 2 columns, got shape {emb.shape!r}."
        logger.error("plot_embedding: %s", msg)
        raise ValueError(msg)

    fig, ax = _get_fig_ax(ax)

    if labels is None:
        labels = ["dim1", "dim2"]

    if y is None:
        logger.debug("plot_embedding: plotting embedding without labels.")
        ax.scatter(emb[:, 0], emb[:, 1], alpha=0.8)
    else:
        y_arr = _ensure_1d(y)
        if len(y_arr) != emb.shape[0]:
            msg = "Length of y must match number of rows in embedding."
            logger.error("plot_embedding: %s", msg)
            raise ValueError(msg)

        unique_labels = np.unique(y_arr)
        logger.debug(
            "plot_embedding: plotting embedding for %d classes.",
            len(unique_labels),
        )
        for lab in unique_labels:
            mask = y_arr == lab
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                alpha=0.8,
                label=str(lab),
            )
        ax.legend(title="label")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title is None:
        title = "2D embedding"
    ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    X: pd.DataFrame,
    *,
    method: str = "pearson",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Axes:
    """Plot a correlation heatmap for numeric columns in a DataFrame.

    Parameters
    ----------
    X :
        Input DataFrame. Only numeric columns are considered.
    method :
        Correlation method passed to ``DataFrame.corr`` (e.g. ``"pearson"``,
        ``"spearman"``, ``"kendall"``).
    ax :
        Optional Matplotlib Axes object. If ``None``, a new figure and axes
        are created.
    title :
        Optional plot title. If omitted, a default title is generated.
    vmin, vmax :
        Value range for the colour scale.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        msg = "No numeric columns found in X for correlation heatmap."
        logger.error("plot_correlation_heatmap: %s", msg)
        raise ValueError(msg)

    logger.debug(
        "plot_correlation_heatmap: computing %s correlations for %d numeric columns.",
        method,
        numeric.shape[1],
    )
    corr = numeric.corr(method=method)

    fig, ax = _get_fig_ax(ax)
    im = ax.imshow(corr.to_numpy(), vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    if title is None:
        title = f"Correlation matrix ({method})"
    ax.set_title(title)

    fig.tight_layout()
    return ax
