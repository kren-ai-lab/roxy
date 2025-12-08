"""Roxy viz package.

This subpackage provides visualisation utilities for exploratory data
analysis and inspection of feature matrices and low-dimensional
embeddings.

It is organised into two layers:

- :mod:`roxy.viz.plots`  : Matplotlib-based, static plotting helpers.
- :mod:`roxy.viz.dashboard` : Optional Plotly-based interactive figures
  and a minimal :class:`RoxyDashboard` container that other frontends
  (Dash, Streamlit, etc.) can embed.
"""

from .plots import (
    plot_feature_distribution,
    plot_feature_boxplot,
    plot_scatter_features,
    plot_embedding,
    plot_correlation_heatmap,
)
from .dashboard import (
    RoxyDashboard,
    interactive_feature_distribution,
    interactive_embedding,
)

__all__ = [
    # static plots
    "plot_feature_distribution",
    "plot_feature_boxplot",
    "plot_scatter_features",
    "plot_embedding",
    "plot_correlation_heatmap",
    # interactive / dashboard
    "RoxyDashboard",
    "interactive_feature_distribution",
    "interactive_embedding",
]
